import copy
import enum
from dataclasses import dataclass
from typing import Sequence, Tuple, Any, NamedTuple

import numpy as np
import torch
from tensordict import TensorDict
from torch import optim, Tensor
from torch.utils._pytree import tree_map


class EntropySchedule:
    """An increasing list of steps where the regularisation network is updated.

    Example
    EntropySchedule([3, 5, 10], [2, 4, 1])
    =>   [0, 3, 6, 11, 16, 21, 26, 36]
          | 3 x2 |      5 x4     | 10 x1
    """

    def __init__(self, *, sizes: Sequence[int], repeats: Sequence[int]):
        """Constructs a schedule of entropy iterations.

            Args:
              sizes: the list of iteration sizes.
              repeats: the list, parallel to sizes, with the number of times for each
                size from `sizes` to repeat.
            """
        try:
            if len(repeats) != len(sizes):
                raise ValueError("`repeats` must be parallel to `sizes`.")
            if not sizes:
                raise ValueError("`sizes` and `repeats` must not be empty.")
            if any([(repeat <= 0) for repeat in repeats]):
                raise ValueError("All repeat values must be strictly positive")
            if repeats[-1] != 1:
                raise ValueError("The last value in `repeats` must be equal to 1, "
                                 "since the last iteration size is repeated forever.")
        except ValueError as e:
            raise ValueError(
                f"Entropy iteration schedule: repeats ({repeats}) and sizes"
                f" ({sizes})."
            ) from e

        schedule = [0]
        for size, repeat in zip(sizes, repeats):
            schedule.extend([schedule[-1] + (i + 1) * size for i in range(repeat)])

        self.schedule = np.array(schedule, dtype=np.int32)

    def __call__(self, learner_step: int) -> Tuple[float, bool]:
        """Entropy scheduling parameters for a given `learner_step`.

            Args:
              learner_step: The current learning step.

            Returns:
              alpha: The mixing weight (from [0, 1]) of the previous policy with
                the one before for computing the intrinsic reward.
              update_target_net: A boolean indicator for updating the target network
                with the current network.
            """

        # The complexity below is because at some point we might go past
        # the explicit schedule, and then we'd need to just use the last step
        # in the schedule and apply the logic of
        # ((learner_step - last_step) % last_iteration) == 0)

        # The schedule might look like this:
        # X----X-------X--X--X--X--------X
        # learner_step | might be here ^    |
        # or there     ^                    |
        # or even past the schedule         ^

        # We need to deal with two cases below.
        # Instead of going for the complicated conditional, let's just
        # compute both and then do the A * s + B * (1 - s) with s being a bool
        # selector between A and B.

        # 1. assume learner_step is past the schedule,
        #    ie schedule[-1] <= learner_step.
        last_size = self.schedule[-1] - self.schedule[-2]
        last_start = self.schedule[-1] + (
                learner_step - self.schedule[-1]) // last_size * last_size
        # 2. assume learner_step is within the schedule.
        start = np.amax(self.schedule * (self.schedule <= learner_step))
        finish = np.amin(self.schedule * (learner_step < self.schedule),
                         initial=self.schedule[-1].item(),
                         where=(learner_step < self.schedule))
        size = finish - start

        # Now select between the two.
        beyond = (self.schedule[-1] <= learner_step)  # Are we past the schedule?
        iteration_start = (last_start * beyond + start * (1 - beyond))
        iteration_size = (last_size * beyond + size * (1 - beyond))

        update_target_net = np.logical_and(
            learner_step > 0,
            np.sum(learner_step == iteration_start + iteration_size - 1),
        )
        alpha = np.minimum(
            (2.0 * (learner_step - iteration_start)) / iteration_size, 1.0)

        return alpha, update_target_net

def _policy_ratio(pi: torch.Tensor, mu: torch.Tensor, actions_oh: torch.Tensor,
                  valid: torch.Tensor) -> torch.Tensor:
    """Returns a ratio of policy pi/mu when selecting action a.

    By convention, this ratio is 1 on non valid states
    Args:
    pi: the policy of shape [..., A].
    mu: the sampling policy of shape [..., A].
    actions_oh: a one-hot encoding of the current actions of shape [..., A].
    valid: 0 if the state is not valid and else 1 of shape [...].

    Returns:
    pi/mu on valid states and 1 otherwise. The shape is the same
    as pi, mu or actions_oh but without the last dimension A.
    """

    def _select_action_prob(pi):
        return (torch.sum(actions_oh * pi, dim=-1, keepdim=False) * valid.float() +
                (1 - valid.float()))

    pi_actions_prob = _select_action_prob(pi)
    mu_actions_prob = _select_action_prob(mu)
    return pi_actions_prob / mu_actions_prob

def _where(pred: torch.Tensor, true_data: Any, false_data: Any) -> Any:
    """
    Similar to jax.where but treats `pred` as a broadcastable prefix.
    Applies the conditional operation element-wise over a tree structure.

    Args:
        pred (torch.Tensor): A boolean tensor used as the condition.
        true_data (Any): A PyTorch tensor or tree of tensors for the "True" branch.
        false_data (Any): A PyTorch tensor or tree of tensors for the "False" branch.

    Returns:
        Any: A PyTorch tensor or tree of tensors with the same structure as true_data/false_data.
    """
    def _where_one(t, f):
        if t.ndim != f.ndim:
            raise ValueError(f"Tensors must have the same rank: {t.ndim} vs {f.ndim}")

        # Ensure ranks match by reshaping `pred`
        pred_reshaped = pred.view(pred.shape + (1,) * (t.ndim - pred.ndim))
        return torch.where(pred_reshaped, t, f)


    return tree_map(_where_one, true_data, false_data)

def v_trace(
    v: torch.Tensor,
    merged_policy: torch.Tensor,
    merged_log_policy: torch.Tensor,
    player: int,
    batch: TensorDict,
    # Scalars below.
    eta: float,
    lambda_: float,
    c: float,
    rho: float,
) -> Tuple[Any, Any, Any]:
    valid = batch["collector"]["mask"]
    assert valid.shape == batch.shape
    player_id = batch["cur_player"]
    assert  player_id.shape == batch.shape
    acting_policy = batch["policy"]
    assert acting_policy.shape == batch.shape + (16,)
    player_others: torch.Tensor = torch.tensor(2 * valid * (batch["cur_player"] == player) - 1).unsqueeze(-1)
    actions_oh = batch["action_one_hot"]
    assert actions_oh.shape == batch.shape + (16,)
    reward = batch["next"]["reward"].squeeze(-1) * batch["cur_player"] * player

    gamma = 1.0

    has_played = valid * (player_id == player)

    policy_ratio = _policy_ratio(merged_policy, acting_policy, actions_oh, valid)
    inv_mu = _policy_ratio(torch.ones_like(merged_policy), acting_policy, actions_oh, valid)

    eta_reg_entropy = (-eta * torch.sum(merged_policy * merged_log_policy, dim=-1) * torch.squeeze(player_others, dim=-1))
    eta_log_policy = -eta * merged_log_policy * player_others

    class LoopVTraceCarry(NamedTuple):
        """The carry of the v-trace scan loop."""
        reward: torch.Tensor
        reward_uncorrected: torch.Tensor
        next_value: torch.Tensor
        next_v_target: torch.Tensor
        importance_sampling: torch.Tensor

    # Initialize the state
    init_state_v_trace = LoopVTraceCarry(
        reward=torch.zeros_like(reward[-1]),
        reward_uncorrected=torch.zeros_like(reward[-1]),
        next_value=torch.zeros_like(v[-1]),
        next_v_target=torch.zeros_like(v[-1]),
        importance_sampling=torch.ones_like(policy_ratio[-1]),
    )

    def _loop_v_trace(carry: LoopVTraceCarry,
                      x: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
                      ) -> Tuple[LoopVTraceCarry, Any]:
        cs, player_id, v, reward, eta_reg_entropy, valid, inv_mu, actions_oh, eta_log_policy = x

        reward_uncorrected = reward + gamma * carry.reward_uncorrected + eta_reg_entropy
        discounted_reward = reward + gamma * carry.reward

        # V-target:
        our_v_target = (
                v + torch.minimum(torch.tensor(rho), cs * carry.importance_sampling).unsqueeze(-1) *
                (reward_uncorrected.unsqueeze(-1) + gamma * carry.next_value - v) +
                lambda_ * torch.minimum(torch.tensor(c), cs * carry.importance_sampling).unsqueeze(-1) * gamma *
                (carry.next_v_target - carry.next_value)
        )

        opp_v_target = torch.zeros_like(our_v_target)
        reset_v_target = torch.zeros_like(our_v_target)

        # Learning output:
        our_learning_output = (
                v +  # value
                eta_log_policy +  # regularization
                actions_oh * inv_mu.unsqueeze(-1) *
                (discounted_reward.unsqueeze(-1) + gamma * carry.importance_sampling.unsqueeze(-1) *
                 carry.next_v_target - v)
        )

        opp_learning_output = torch.zeros_like(our_learning_output)
        reset_learning_output = torch.zeros_like(our_learning_output)

        # State carry:
        our_carry = LoopVTraceCarry(
            reward=torch.zeros_like(carry.reward),
            next_value=v,
            next_v_target=our_v_target,
            reward_uncorrected=torch.zeros_like(carry.reward_uncorrected),
            importance_sampling=torch.ones_like(carry.importance_sampling),
        )
        opp_carry = LoopVTraceCarry(
            reward=eta_reg_entropy + cs * discounted_reward,
            reward_uncorrected=reward_uncorrected,
            next_value=gamma * carry.next_value,
            next_v_target=gamma * carry.next_v_target,
            importance_sampling=cs * carry.importance_sampling,
        )
        reset_carry = init_state_v_trace

        # Invalid turn: reset state
        condition_valid = valid.bool()
        condition_player = (player_id == player).bool()

        result_carry, result_outputs = _where(
            condition_valid,
            _where(
                condition_player,
                (our_carry, (our_v_target, our_learning_output)),
                (opp_carry, (opp_v_target, opp_learning_output)),
            ),
            (reset_carry, (reset_v_target, reset_learning_output))
        )

        return result_carry, result_outputs

    # Apply scan using PyTorch's loop (manual lax.scan)
    v_target_list, learning_output_list = [], []
    carry = init_state_v_trace

    for i in reversed(range(policy_ratio.shape[0])):
        carry, (v_target, learning_output) = _loop_v_trace(
            carry, (
            policy_ratio[i], player_id[i], v[i], reward[i], eta_reg_entropy[i], valid[i], inv_mu[i], actions_oh[i],
            eta_log_policy[i])
        )
        v_target_list.append(v_target)
        learning_output_list.append(learning_output)

    # Reverse the results to match original order
    v_target = torch.stack(v_target_list[::-1])
    learning_output = torch.stack(learning_output_list[::-1])

    # Return the final results
    return v_target, has_played, learning_output

def get_loss_v(v_list: Sequence[Tensor],
               v_target_list: Sequence[Tensor],
               mask_list: Sequence[Tensor]) -> Tensor:
    """Define the loss function for the critic."""
    # v_list and v_target_list come with a degenerate trailing dimension,
    # which mask_list tensors do not have.
    loss_v_list = []
    for (v_n, v_target, mask) in zip(v_list, v_target_list, mask_list):
        assert v_n.shape[0] == v_target.shape[0]

        loss_v = torch.unsqueeze(mask, dim=-1) * (v_n - v_target.detach())**2
        normalization = torch.sum(mask)
        loss_v = torch.sum(loss_v) / (normalization + (normalization == 0.0))

        loss_v_list.append(loss_v)
    return sum(loss_v_list)


def apply_force_with_threshold(decision_outputs: Tensor, force: Tensor,
                               threshold: float,
                               threshold_center: Tensor) -> Tensor:
    """Apply the force with below a given threshold."""
    can_decrease = decision_outputs - threshold_center > -threshold
    can_increase = decision_outputs - threshold_center < threshold
    force_negative = torch.minimum(force, torch.tensor(0.0))
    force_positive = torch.maximum(force, torch.tensor(0.0))
    clipped_force = can_decrease * force_negative + can_increase * force_positive
    return decision_outputs * clipped_force.detach()


def renormalize(loss: Tensor, mask: Tensor) -> Tensor:
    """The `normalization` is the number of steps over which loss is computed."""
    loss = torch.sum(torch.where(mask != 0, loss, torch.zeros_like(loss)))
    normalization = torch.sum(mask)
    return loss / (normalization + (normalization == 0.0))


def get_loss_nerd(logit_list: Sequence[Tensor],
                  policy_list: Sequence[Tensor],
                  q_vr_list: Sequence[Tensor],
                  valid: Tensor,
                  player_ids: Sequence[Tensor],
                  legal_actions: Tensor,
                  importance_sampling_correction: Sequence[Tensor],
                  clip: float = 100,
                  threshold: float = 2) -> Tensor:
    """Define the nerd loss."""
    assert isinstance(importance_sampling_correction, list)
    loss_pi_list = []
    num_valid_actions = torch.sum(legal_actions, dim=-1, keepdim=True)
    for (k, logit_pi, pi, q_vr, is_c) in zip([1, -1], logit_list, policy_list, q_vr_list, importance_sampling_correction):
        assert logit_pi.shape[0] == q_vr.shape[0]
        # loss policy
        adv_pi = q_vr - torch.sum(pi * q_vr, dim=-1, keepdim=True)
        adv_pi = is_c * adv_pi  # importance sampling correction
        adv_pi = torch.clip(adv_pi, min=-clip, max=clip)
        adv_pi = adv_pi.detach()

        valid_logit_sum = torch.sum(logit_pi * legal_actions, dim=-1, keepdim=True)
        mean_logit = valid_logit_sum / (num_valid_actions + (num_valid_actions == 0))

        # Subtract only the mean of the valid logits
        logits = logit_pi - mean_logit

        threshold_center = torch.zeros_like(logits)

        nerd_loss = torch.sum(
            legal_actions *
            apply_force_with_threshold(logits, adv_pi, threshold, threshold_center),
            dim=-1)
        nerd_loss = -renormalize(nerd_loss, valid * (player_ids == k))
        loss_pi_list.append(nerd_loss)

    return sum(loss_pi_list)
