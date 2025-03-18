
import itertools
import numpy as np
import pytest

from stratego_gym.envs.primitives import Player
from stratego_gym.envs.startego import StrategoEnv, GamePhase

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
    "red,blue",
    itertools.product(
        [True, False],
        [True, False],
    )
)
def test_lose_on_deploy(env_original: StrategoEnv, red: bool, blue: bool):
    first_row = 6  # front line at deployment
    terminated = False
    lakes_mask = env_original.lakes.sum(axis=0).astype(bool)
    lakes_mask = np.arange(env_original.width)[~lakes_mask]
    for i in range(env_original.config.p1_pieces_num.sum() + env_original.config.p2_pieces_num.sum()):
        assert env_original.game_phase == GamePhase.DEPLOY
        assert env_original.player == Player.RED if i % 2 == 0 else Player.BLUE
        if i < 2 * len(lakes_mask):
            if i % 2 == 0 and red or i % 2 != 0 and blue:
                state, reward, terminated, truncated, info = env_original.step((first_row, lakes_mask[i // 2]))
                continue

        action = env_original.action_space.sample()
        state, reward, terminated, truncated, info = env_original.step(action)

    if red and blue:
        assert env_original.game_phase == GamePhase.TERMINAL
        assert reward == 0
    elif red:
        if env_original.valid_pieces_to_select(is_other_player=True).sum() > 0:
            assert reward == -1
        else:
            assert reward == 0
    elif blue:
        if env_original.valid_pieces_to_select(is_other_player=True).sum() > 0:
            assert reward == 1
        else:
            assert reward == 0

        