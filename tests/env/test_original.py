import itertools
from typing import Any, Callable
import numpy as np
import pytest

from stratego.core.primitives import Player
from stratego.core.stratego import StrategoEnv, GamePhase


@pytest.mark.parametrize("compile", [True, False])
def test_single_trajectory_rollout(
    env_original: Callable[[Any], StrategoEnv], compile: bool
):
    env = env_original(compile)
    terminated = False
    for i in range(80):
        assert env.game_phase == GamePhase.DEPLOY
        assert env.player == Player.RED if i % 2 == 0 else Player.BLUE
        action = env.action_space.sample()
        print(env.action_space.mask)

        state, reward, terminated, truncated, info = env.step(action)

    # assert env.game_phase == GamePhase.SELECT
    # assert env.player == Player.RED
    # while not terminated:
    #     action = env.action_space.sample()
    #     state, reward, terminated, truncated, info = env.step(action)

    # assert env.game_phase == GamePhase.TERMINAL


@pytest.mark.parametrize(
    "compile,red_surrounded,blue_surrounded",
    itertools.product(
        [True, False],
        [True, False],
        [True, False],
    ),
)
def test_lose_on_deploy(
    env_original: Callable[[Any], StrategoEnv],
    compile: bool,
    red_surrounded: bool,
    blue_surrounded: bool,
):
    env = env_original(compile)
    first_row = 6  # front line at deployment
    terminated = False
    lakes_mask = env.lakes.sum(axis=0).astype(bool)
    lakes_mask = np.arange(env.width)[~lakes_mask]
    for i in range(80):
        assert env.game_phase == GamePhase.DEPLOY
        assert env.player == Player.RED if i % 2 == 0 else Player.BLUE
        if i < 2 * len(lakes_mask):
            if i % 2 == 0 and red_surrounded or i % 2 != 0 and blue_surrounded:
                state, reward, terminated, truncated, info = env.step(
                    (first_row, lakes_mask[i // 2])
                )
                continue

        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)

    assert env.game_phase != GamePhase.DEPLOY
    if red_surrounded and blue_surrounded or red_surrounded:
        assert env.game_phase == GamePhase.TERMINAL
        # If the game ends with blue's turn, but he can move, he is awarded 1 on his turn.
        assert (reward == 0) if blue_surrounded else (reward == 1)
    elif blue_surrounded:
        assert env.game_phase == GamePhase.SELECT
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.game_phase == GamePhase.TERMINAL
        assert reward == 1
    else:
        # not red_surrounded and not blue_surrounded
        pass
