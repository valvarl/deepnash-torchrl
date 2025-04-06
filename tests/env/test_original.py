
import itertools
import numpy as np
import pytest

from stratego.core.primitives import Player
from stratego.core.startego import StrategoEnv, GamePhase

def test_single_trajectory_rollout(env_original: StrategoEnv):
    terminated = False
    for i in range(env_original.config.p1_pieces_num.sum() + env_original.config.p2_pieces_num.sum()):
        assert env_original.game_phase == GamePhase.DEPLOY
        assert env_original.player == Player.RED if i % 2 == 0 else Player.BLUE
        action = env_original.action_space.sample()
        state, reward, terminated, truncated, info = env_original.step(action)

    assert env_original.game_phase == GamePhase.SELECT
    assert env_original.player == Player.RED
    while not terminated:
        action = env_original.action_space.sample()
        state, reward, terminated, truncated, info = env_original.step(action)
    
    assert env_original.game_phase == GamePhase.TERMINAL


@pytest.mark.parametrize(
    "red_surrounded,blue_surrounded",
    itertools.product(
        [True, False],
        [True, False],
    )
)
def test_lose_on_deploy(env_original: StrategoEnv, red_surrounded: bool, blue_surrounded: bool):
    first_row = 6  # front line at deployment
    terminated = False
    lakes_mask = env_original.lakes.sum(axis=0).astype(bool)
    lakes_mask = np.arange(env_original.width)[~lakes_mask]
    for i in range(env_original.config.p1_pieces_num.sum() + env_original.config.p2_pieces_num.sum()):
        assert env_original.game_phase == GamePhase.DEPLOY
        assert env_original.player == Player.RED if i % 2 == 0 else Player.BLUE
        if i < 2 * len(lakes_mask):
            if i % 2 == 0 and red_surrounded or i % 2 != 0 and blue_surrounded:
                state, reward, terminated, truncated, info = env_original.step((first_row, lakes_mask[i // 2]))
                continue

        action = env_original.action_space.sample()
        state, reward, terminated, truncated, info = env_original.step(action)

    assert env_original.game_phase != GamePhase.DEPLOY
    if red_surrounded and blue_surrounded or red_surrounded:
        assert env_original.game_phase == GamePhase.TERMINAL
        # If the game ends with blue's turn, but he can move, he is awarded 1 on his turn.
        assert (reward == 0) if blue_surrounded else (reward == 1)
    elif blue_surrounded:
        assert env_original.game_phase == GamePhase.SELECT
        state, reward, terminated, truncated, info = env_original.step(env_original.action_space.sample())
        state, reward, terminated, truncated, info = env_original.step(env_original.action_space.sample())
        assert env_original.game_phase == GamePhase.TERMINAL
        assert reward == 1
    else:
        # not red_surrounded and not blue_surrounded
        pass
        