import typing as tp

import numpy as np
from gymnasium import spaces

from stratego.core.config import GameMode
from stratego.core.masked_multi_descrete import MaskedMultiDiscrete
from stratego.core.primitives import Player
from stratego.core.stratego import GamePhase, StrategoEnvBase
from stratego.wrappers.cpp_config import StrategoConfigCpp

from stratego.cpp import stratego_cpp as sp


class StrategoEnvCpp(StrategoEnvBase):
    def __init__(
        self, config: StrategoConfigCpp | None = None, render_mode: str | None = None
    ):
        self._config = config
        if config is None:
            self._config = StrategoConfigCpp.from_game_mode(GameMode.ORIGINAL)

        self._env_cpp = sp.StrategoEnv(self._config._config_cpp)

        self.observation_space = self._get_observation_space()
        self.action_space: MaskedMultiDiscrete = self._get_action_space()

    def _get_observation_space(self):
        observation_channels = (
            len(self._config.allowed_pieces) * 3
            + self._config.observed_history_entries
            + 6
        )
        shape = (observation_channels, self._config.height, self._config.width)
        mask_shape = (self._config.height, self._config.width)

        return spaces.Dict(
            {
                "obs": spaces.Box(low=-3, high=1, shape=shape, dtype=np.float64),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=mask_shape, dtype=np.int64
                ),
            }
        )

    def _get_action_space(self):
        if getattr(self, "action_space", None) is None:
            return MaskedMultiDiscrete((self.height, self.width), dtype=np.int64)
        else:
            return self.action_space

    def reset(self, seed=None, options=None):
        if seed is None:
            seed = 35

        super().reset(seed=seed, options=options)

        if (
            getattr(self, "height", None) != self._config.height
            or getattr(self, "width", None) != self._config.width
        ):
            del self.action_space
            self.height = self._config.height
            self.width = self._config.width
            self.action_space = self._get_action_space()

        (
            obs,
            action_mask,
        ) = self._env_cpp.reset(seed)
        return (
            {"obs": obs, "action_mask": action_mask},
            None,
        )

    def step(self, action):
        obs, action_mask, reward, terminated, truncated = self._env_cpp.step(action)
        return (
            {"obs": obs, "action_mask": action_mask},
            reward,
            terminated,
            truncated,
            None,
        )

    def get_info(self) -> dict[str : tp.Any]:
        return self._env_cpp.get_info()

    @property
    def config(self):
        return self._config

    @property
    def height(self) -> int:
        return self._env_cpp.height

    @property
    def width(self) -> int:
        return self._env_cpp.width

    @property
    def game_phase(self) -> GamePhase:
        return GamePhase(self._env_cpp.game_phase.value)

    @property
    def player(self) -> Player:
        return Player(self._env_cpp.current_player.value)

    @property
    def board(self) -> np.ndarray:
        return self._env_cpp.board

    @property
    def lakes(self) -> np.ndarray:
        return self._env_cpp.lakes
