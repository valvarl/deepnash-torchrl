# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Efficient Trajectory Sampling with CompletedTrajRepertoire

This example demonstrates how to design a custom transform that filters trajectories during sampling,
ensuring that only completed trajectories are present in sampled batches. This can be particularly useful
when dealing with environments where some trajectories might be corrupted or never reach a done state,
which could skew the learning process or lead to biased models. For instance, in robotics or autonomous
driving, a trajectory might be interrupted due to external factors such as hardware failures or human
intervention, resulting in incomplete or inconsistent data. By filtering out these incomplete trajectories,
we can improve the quality of the training data and increase the robustness of our models.
"""

import torch
from tensordict import TensorDictBase
from torchrl.data import LazyTensorStorage, ReplayBuffer, BoundedTensorSpec
from torchrl.envs import GymEnv, TrajCounter, Transform


class CompletedTrajectoryRepertoire(Transform):
    """
    A transform that keeps track of completed trajectories and filters them out during sampling.
    """

    def __init__(self):
        super().__init__()
        self.completed_trajectories = set()
        self.repertoire_tensor = torch.zeros((), dtype=torch.int64)

    def _update_repertoire(self, tensordict: TensorDictBase) -> None:
        """Updates the repertoire of completed trajectories."""
        done = tensordict["next", "terminated"].squeeze(-1)
        traj = tensordict["next", "traj_count"][done].view(-1)
        if traj.numel():
            self.completed_trajectories = self.completed_trajectories.union(
                traj.tolist()
            )
            self.repertoire_tensor = torch.tensor(
                list(self.completed_trajectories), dtype=torch.int64
            )

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Updates the repertoire of completed trajectories during insertion."""
        self._update_repertoire(tensordict)
        return tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Filters out incomplete trajectories during sampling."""
        traj = tensordict["next", "traj_count"]
        traj = traj.unsqueeze(-1)
        has_traj = (traj == self.repertoire_tensor).any(-1)
        has_traj = has_traj.view(tensordict.shape)
        return tensordict[has_traj]


class QuantizeTransform(Transform):
    """
    -(2 + t / 12) -> 0..11
    -1 -> 12
    0..1 -> 13..255
    """

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        # Prepare bin edges for special values
        t_vals = torch.arange(12, dtype=obs.dtype, device=obs.device)
        special_values = -(2 + t_vals / 12)
        bin_edges = torch.cat(
            [special_values, torch.tensor([-1.0], dtype=obs.dtype, device=obs.device)]
        )
        bin_values = torch.arange(13, dtype=torch.uint8, device=obs.device)  # 0..12

        obs_flat = obs.flatten()
        quantized = torch.full_like(obs_flat, fill_value=13, dtype=torch.uint8)

        # Bucketize obs: which bin it falls into (exact match)
        for i, val in enumerate(bin_edges):
            quantized[obs_flat == val] = bin_values[i]

        # For values in [0..1] -> scaled to 13..255
        linear_mask = (obs_flat >= 0.0) & (obs_flat <= 1.0)
        quantized[linear_mask] = (
            (obs_flat[linear_mask] * (255 - 13) + 13).round().to(torch.uint8)
        )

        return quantized.view_as(obs)

    # The transform must also modify the data at reset time
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)


class DequantizeTransform(Transform):
    """
    0..11 -> -(2 + t / 12)
    12 -> -1
    13..255 -> 0..1
    """

    def _apply_transform(self, obs_qunat):
        obs = torch.empty_like(obs_qunat, dtype=torch.float32)

        # 0..11 → -(2 + t/12)
        for t in range(12):
            obs[obs_qunat == t] = -(2 + t / 12)

        # 12 → -1.0
        obs[obs_qunat == 12] = -1.0

        # 13..255 → обратно в диапазон 0..1
        mask = obs_qunat >= 13
        obs[mask] = (obs_qunat[mask] - 13) / (255 - 13)

        return obs

    # The transform must also modify the data at reset time
    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)
