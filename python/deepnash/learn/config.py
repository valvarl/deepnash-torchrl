
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


@dataclass(frozen=True)
class AdamConfig:
    """Adam optimizer related params."""
    b1: float = 0.0
    b2: float = 0.999
    eps: float = 10e-8

@dataclass(frozen=True)
class NerdConfig:
    """Nerd related params."""
    beta: float = 2.0
    clip: float = 10_000

class StateRepresentation(str, Enum):
    INFO_SET = "info_set"
    OBSERVATION = "observation"

@dataclass(frozen=True)
class RNaDConfig:
    """Configuration parameters for the RNaDSolver."""
    # The game parameter string including its name and parameters.
    game_name: str
    # The games longer than this value are truncated. Must be strictly positive.
    trajectory_max: int = 10

    # The content of the EnvStep.obs tensor.
    state_representation: StateRepresentation = StateRepresentation.INFO_SET

    # Network configuration.
    policy_network_layers: Sequence[int] = (256, 256)

    # The batch size to use when learning/improving parameters.
    batch_size: int = 256
    # The learning rate for `params`.
    learning_rate: float = 0.0005
    # The config related to the ADAM optimizer used for updating `params`.
    adam: AdamConfig = AdamConfig()
    # All gradients values are clipped to [-clip_gradient, clip_gradient].
    clip_gradient: float = 10_000
    # The "speed" at which `params_target` is following `params`.
    target_network_avg: float = 0.001

    # RNaD algorithm configuration.
    # Entropy schedule configuration. See EntropySchedule class documentation.
    # entropy_schedule_repeats: Sequence[int] = (100, 65, 34, 1)
    # entropy_schedule_size: Sequence[int] = (10_000, 100_000, 35_000, 35_000)
    entropy_schedule_repeats: Sequence[int] = (200,1,)
    entropy_schedule_size: Sequence[int] = (100,100,)
    # The weight of the reward regularisation term in RNaD.
    eta_reward_transform: float = 0.2
    nerd: NerdConfig = NerdConfig()
    c_vtrace: float = 1.0

    # Options related to fine tuning of the agent.
    # finetune: FineTuning = FineTuning() # TODO Add this back in

    # The seed that fully controls the randomness.
    seed: int = 42
