from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Sequence
from omegaconf import DictConfig
from tensordict.nn import TensorDictModule

from deepnash.agents.registry import build_module


@dataclass(frozen=True)
class DeepNashAgentConfig(ABC):

    class_name: str = "DeepNashAgent"

    def __post_init__(self):
        if self.__class__ == DeepNashAgentConfig:
            raise TypeError("Cannot instantiate abstract class.")

    @staticmethod
    def from_config(agent_config: DictConfig) -> DeepNashAgentConfig:
        class_name = agent_config.class_name
        return build_module(class_name + "Config", **agent_config)


class DeepNashAgent(TensorDictModule, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_config(config: DeepNashAgentConfig) -> DeepNashAgent:
        return build_module(config.class_name, config)


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
    entropy_schedule_repeats: Sequence[int] = (
        200,
        1,
    )
    entropy_schedule_size: Sequence[int] = (
        100,
        100,
    )
    # The weight of the reward regularisation term in RNaD.
    eta_reward_transform: float = 0.2
    nerd: NerdConfig = NerdConfig()
    c_vtrace: float = 1.0

    # Options related to fine tuning of the agent.
    # finetune: FineTuning = FineTuning() # TODO Add this back in

    # The seed that fully controls the randomness.
    seed: int = 42


class DeploymentMode(str, Enum):
    """Enum for deployment modes."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


@dataclass(frozen=True)
class DeploymentConfig:
    """Deployment configuration for GPU usage in actor and learner."""

    mode: DeploymentMode = DeploymentMode.PARALLEL
    sequential_actor_learner_ratio: float = 1 / 4
    sequential_step_duration: int = 300

    num_instances: int = 1
    instance_gpu_capacity: int = 2

    actor_gpu_count: int = 1
    learner_gpu_count: int = 1

    shared_replay_buffer: bool = False


@dataclass(frozen=True)
class SyncIntervals:
    intra_instance: int = 5
    inter_instance: int = 20


@dataclass(frozen=True)
class EventConfig:
    wandb: bool = True

    log_interval: int = 5

    eval_interval: int = 20

    checkpoint_interval: int = 20

    sync_intervals: SyncIntervals = SyncIntervals()
    actor_update_interval: int = 5


class CollectorType(str, Enum):

    SYNC = "sync"


@dataclass(frozen=True)
class CollectorConfig:
    collector_type: CollectorType = CollectorType.SYNC

    n_workers: int = 1
    frames_per_batch: int = 3600
    max_frames_per_traj: int = 3600
    total_frames: int = -1
    device: str = "cuda"
    policy_device: str = "cuda"
    env_device: str = "cpu"
    storing_device: str = "cuda"
    split_trajs: bool = True
    reset_at_each_iter: bool = False

    def validate(self):
        if self.collector_type == CollectorType.SYNC:
            assert self.n_workers > 0, "n_workers must be greater than 0."
            assert self.frames_per_batch > 0, "frames_per_batch must be greater than 0."
            assert (
                self.max_frames_per_traj > 0
            ), "max_frames_per_traj must be greater than 0."
            assert self.total_frames > 0, (
                "total_frames must be greater than 0. Use -1 for infinite."
                if self.total_frames != -1
                else True
            )
        else:
            raise ValueError(f"Collector type {self.collector_type} is not supported.")


class ReplayBufferType(str, Enum):

    CONSTANT_LENGTH = "constant_length"
    ENSEMBLE = "ensemble"


@dataclass(frozen=True)
class ReplayBufferConfig:
    buffer_type: ReplayBufferType = ReplayBufferType.ENSEMBLE

    save_dir: str = "data"

    buffer_capacity: int = 1000

    max_trajectory_length: int = 3600

    buffer_lengths: Sequence[int] | None = None
    min_length: int = 200
    max_length: int = 3600
    step: int = 200

    prioritized_replay: bool = False
    obs_quantization: bool = False

    def validate(self):
        if self.buffer_type == ReplayBufferType.ENSEMBLE:
            if self.buffer_lengths is None:
                self.buffer_lengths = list(
                    range(self.min_length, self.max_length + 1, self.step)
                )
            else:
                assert (
                    len(self.buffer_lengths) > 0
                ), "buffer_lengths must be a non-empty list."
                assert all(
                    isinstance(x, int) for x in self.buffer_lengths
                ), "buffer_lengths must be a list of integers."
                assert (
                    sorted(self.buffer_lengths) == self.buffer_lengths
                ), "buffer_lengths must be a sorted list of integers."
        elif self.buffer_type == ReplayBufferType.CONSTANT_LENGTH:
            if self.buffer_lengths is not None:
                raise ValueError(
                    "buffer_lengths should be None for constant length replay buffer."
                )
            if self.min_length != self.max_length:
                raise ValueError(
                    "min_length and max_length should be equal for constant length replay buffer."
                )
            if self.step != 1:
                raise ValueError("step should be 1 for constant length replay buffer.")
        else:
            raise ValueError(f"Replay buffer type {self.buffer_type} is not supported.")


class TrainingConfig:
    """Configuration for training."""

    def __init__(
        self,
        agent_config: DeepNashAgentConfig,
        rnad_config: RNaDConfig,
        deployment_config: DeploymentConfig,
        event_config: EventConfig,
        collector_config: CollectorConfig,
        replay_buffer_config: ReplayBufferConfig,
        directory_name: str | None = None,
        use_same_init_net_as: bool = False,
    ):
        self.agent_config = agent_config
        self.rnad_config = rnad_config
        self.deployment_config = deployment_config
        self.event_config = event_config
        self.collector_config = collector_config
        self.replay_buffer_config = replay_buffer_config

        self.directory_name = directory_name
        self.use_same_init_net_as = use_same_init_net_as

    def __repr__(self):
        return (
            f"TrainingConfig(agent_config={self.agent_config}, "
            + f"rnad_config={self.rnad_config}, deployment_config={self.deployment_config}, "
            + f"event_config={self.event_config}, collector_config={self.collector_config}, "
            + f"replay_buffer_config={self.replay_buffer_config}, directory_name={self.directory_name}, "
            + f"use_same_init_net_as={self.use_same_init_net_as})"
        )

    @classmethod
    def from_config(
        cls,
        agent_config: DictConfig,
        training_config: DictConfig,
    ) -> TrainingConfig:
        agent_config = DeepNashAgentConfig.from_config(agent_config)
        rnad_config = RNaDConfig(**training_config.rnad_config)
        deployment_config = DeploymentConfig(**training_config.deployment_config)
        event_config = EventConfig(**training_config.event_config)
        collector_config = CollectorConfig(**training_config.collector_config)
        replay_buffer_config = ReplayBufferConfig(
            **training_config.replay_buffer_config
        )
        directory_name = (
            training_config.directory_name
            if "directory_name" in training_config
            else None
        )
        use_same_init_net_as = (
            training_config.use_same_init_net_as
            if "use_same_init_net_as" in training_config
            else False
        )

        return cls(
            agent_config,
            rnad_config,
            deployment_config,
            event_config,
            collector_config,
            replay_buffer_config,
            directory_name,
            use_same_init_net_as,
        )

    def validate(self):
        self.collector_config.validate()
        self.replay_buffer_config.validate()
