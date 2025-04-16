from deepnash.resources.transforms import (
    CompletedTrajectoryRepertoire,
    DequantizeTransform,
)
import tensordict
import torch
import random
from torchrl.data.replay_buffers import (
    TensorDictReplayBuffer as TDRB,
    LazyTensorStorage,
    PrioritizedSampler,
    ReplayBufferEnsemble,
)
from tensordict import TensorDict
from torchrl.envs.transforms import Compose, ExcludeTransform


class CustomReplayBufferEnsemble(ReplayBufferEnsemble):
    def __init__(
        self,
        buffer_lengths=None,
        buffer_size=1000,
        num_buffer_sampled: int | None = None,
        **kwargs,
    ):
        """
        Args:
            buffer_lengths (list of int): list of trajectory lengths for the sub-buffers.
                By default, generates a list from 200 to 3600 with step 200.
            buffer_size (int): capacity for each sub-buffer (assumed constant for now).
            kwargs: additional keyword arguments passed to the sub-buffers.
        """
        if buffer_lengths is None:
            # Default trajectory lengths: from 200 to 3600 (inclusive) with step 200.
            buffer_lengths = list(range(200, 3600 + 1, 200))
        self.buffer_ids = list(
            range(len(buffer_lengths))
        )  # Assign a unique id to each sub-buffer
        self.buffer_lengths = buffer_lengths

        # Create individual sub-buffers with specified capacities and trajectory lengths.
        buffers = []
        for length in buffer_lengths:
            # Create a storage and sampler for each sub-buffer.
            storage = LazyTensorStorage(buffer_size, ndim=1)
            sampler = PrioritizedSampler(max_capacity=buffer_size, alpha=1.0, beta=1.0)
            transform = Compose(
                # CompletedTrajectoryRepertoire(),
                DequantizeTransform(
                    in_keys=["obs_quant", ("next", "obs_quant")],
                    out_keys=["obs", ("next", "obs")],
                ),
                # ExcludeTransform("obs_quant"),
                # ExcludeTransform(("next", "obs_quant")),
            )
            # Create a custom replay buffer instance with statistics.
            rb = TDRB(
                storage=storage,
                sampler=sampler,
                # transform=transform,
                priority_key="priority",  # key used for prioritized replay updates
                **kwargs,
            )
            # Save the expected trajectory length in a custom attribute
            rb.expected_length = length
            buffers.append(rb)
        # Initialize the ensemble using the list of created sub-buffers.
        super().__init__(*buffers, num_buffer_sampled=num_buffer_sampled, **kwargs)

        self.unfinished_trajectories = {}

    def extend_batch(self, data):
        """
        Extends the replay buffer with trajectories extracted from the input data.

        The function processes each chunk in the data:
        - Uses the 'mask' from data['collector'] to determine the valid length.
        - Skips chunks with zero valid frames or longer than 3600 frames.
        - Checks if the trajectory has finished (using the 'terminated' flag).
        - Restores an unfinished trajectory if available by concatenating with the current chunk.
        - If the trajectory isn't finished and its length is under 3600, stashes it for later completion.
        - When finished, pads the chunk to the required length (multiple of 200) and extends the correct sub-buffer.

        Returns:
            A tuple of statistics:
                (total_frames_added, games_added, games_stashed, games_recovered)
        """
        total_frames_added = 0  # Total number of frames added to sub-buffers.
        games_added = 0  # Count of completed trajectories added.
        games_stashed = 0  # Count of trajectories stashed for future recovery.
        games_recovered = 0  # Count of trajectories recovered from unfinished storage.

        # Compute the valid frame length for each chunk using the provided mask.
        mask_lengths = data["collector"]["mask"].sum(dim=-1)

        # Iterate over each trajectory chunk in the batch.
        for i, chunk in enumerate(data):
            valid_length = mask_lengths[i]
            # Use only the valid portion of the current chunk.
            chunk = chunk[:valid_length]

            # Skip empty chunks or chunks exceeding the maximum allowed length.
            if valid_length == 0 or valid_length > 3600:
                continue

            # Check for termination flag in the 'next' field of the chunk.
            terminated_flags = chunk["next", "terminated"].squeeze(-1)
            finished_traj_ids = chunk["next", "traj_count"][terminated_flags].view(-1)
            # Expect at most one finished trajectory id per chunk.
            assert finished_traj_ids.numel() <= 1
            is_finished = finished_traj_ids.numel() > 0

            # Retrieve the trajectory id from the first element.
            traj_id = int(chunk["traj_count"][0].item())

            # If the trajectory was previously stashed, restore and concatenate it.
            if traj_id in self.unfinished_trajectories:
                chunk = chunk.detach().cpu()
                previous_chunk = self.unfinished_trajectories.pop(traj_id)
                chunk = tensordict.cat([previous_chunk, chunk], dim=0)
                print(
                    f"Restore trajectory {traj_id}, previous shape {previous_chunk.shape}, new chunk shape {chunk.shape}"
                )
                games_recovered += 1

            # If the trajectory is not finished, stash it (if it does not exceed the maximum length).
            if not is_finished:
                if chunk.shape[0] < 3600:
                    self.unfinished_trajectories[traj_id] = chunk.detach().cpu()
                    games_stashed += 1
                continue

            # The trajectory is finished; update overall frames counter.
            total_frames_added += chunk.shape[0]

            # Determine sub-buffer index based on trajectory length; each sub-buffer expects chunks padded to multiples of 200.
            sub_buffer_index = int((chunk.shape[0] + 199) / 200 - 1)
            assert 0 <= sub_buffer_index < 18

            # Pad the chunk to match the exact required number of frames (multiple of 200)
            padded_chunk = tensordict.pad(
                chunk, [0, (sub_buffer_index + 1) * 200 - chunk.shape[0]]
            )[
                None,
            ]
            # Extend the corresponding sub-buffer with the padded chunk.
            self[sub_buffer_index].extend(padded_chunk)
            print(
                f"Trajectory {traj_id} added with shape {padded_chunk.shape} index {sub_buffer_index}"
            )
            games_added += 1

        # Output the list of trajectory IDs that remain unfinished.
        print(list(self.unfinished_trajectories.keys()))
        return (total_frames_added, games_added, games_stashed, games_recovered)
