import itertools
from typing import Any, Callable

import numpy as np
import pytest

from stratego_gym.envs.primitives import Piece, Player, Pos
from stratego_gym.envs.startego import GamePhase, StrategoEnv
from tests.env.utils import (
    SCOUT_ONLY, SPY_ONLY, FLAG_2BOMB_SCOUT, FLAG_2BOMB_SPY,
    move_fwd, repeat_twice, rotate_pos, validate_move
)


@pytest.mark.parametrize(
    "pieces,from_pos,red_surrounded,blue_surrounded",
    itertools.product(
        [SCOUT_ONLY, SPY_ONLY],
        itertools.product(range(3, 5), range(5)),
        [True, False],
        [True, False],
    )
)
def test_lose_on_deploy_surrounded_lakes(
    env_5x5: Callable[[Any], StrategoEnv],
    pieces,
    from_pos: Pos,
    red_surrounded: bool,
    blue_surrounded: bool,
):
    lakes_mask = np.ones((5, 5), dtype=bool)
    lakes_mask[from_pos] = False
    lakes_mask[4 - from_pos[0], 4 - from_pos[1]] = False

    if not red_surrounded:
        lakes_mask[3:] = False
    if not blue_surrounded:
        lakes_mask[:2] = False

    p1_deploy_mask = ~lakes_mask
    p1_deploy_mask[:2] = False
    p2_deploy_mask = ~lakes_mask
    p2_deploy_mask[3:] = False

    env = env_5x5(pieces, lakes_mask=lakes_mask, p1_deploy_mask=p1_deploy_mask, p2_deploy_mask=p2_deploy_mask)
    state, reward, terminated, truncated, info = env.step(from_pos)
    state, reward, terminated, truncated, info = env.step(from_pos)

    assert env.game_phase != GamePhase.DEPLOY
    if red_surrounded and blue_surrounded or red_surrounded:
        assert env.game_phase == GamePhase.TERMINAL
        # if the game ends with blue's turn, but he can move, he is awarded 1 on his turn.
        assert (reward == 0) if blue_surrounded else (reward == 1)
    elif blue_surrounded:
        assert env.game_phase == GamePhase.SELECT
        state, reward, terminated, truncated, info = env.step(from_pos)
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.game_phase == GamePhase.TERMINAL
        assert reward == 1
    else:
        # not red_surrounded and not blue_surrounded
        pass

@pytest.mark.parametrize(
    "pieces,from_pos,red_surrounded,blue_surrounded",
    itertools.product(
        [FLAG_2BOMB_SCOUT, FLAG_2BOMB_SPY],
        itertools.product(range(3, 5), range(5)),
        [True, False],
        [True, False],
    )
)
def test_lose_on_deploy_surrounded_pieces_and_lakes(
    env_5x5: Callable[[Any], StrategoEnv],
    pieces,
    from_pos: Pos,
    red_surrounded: bool,
    blue_surrounded: bool,
):
    lakes_mask = np.ones((5, 5), dtype=bool)
    lakes_mask[3:] = False
    lakes_mask[:2] = False
    p1_deploy_mask = ~lakes_mask
    p1_deploy_mask[:2] = False
    p2_deploy_mask = ~lakes_mask
    p2_deploy_mask[3:] = False

    env = env_5x5(pieces, lakes_mask=lakes_mask, p1_deploy_mask=p1_deploy_mask, p2_deploy_mask=p2_deploy_mask)
    deploy_direction = np.array([-1, 1, 0, 0])
    pieces_deployed = 0
    for direction in zip(deploy_direction, np.roll(deploy_direction, 2)):
        deploy_pos = (from_pos[0] + direction[0], from_pos[1] + direction[1])
        if deploy_pos[0] < 0 or env.height <= deploy_pos[0] or \
            deploy_pos[1] < 0 or env.width <= deploy_pos[1] or \
            env.lakes[deploy_pos]:
            continue
        repeat_twice(env.step, deploy_pos)
        pieces_deployed += 1

    assert pieces_deployed in [2, 3]
    if pieces_deployed == 2:
        deploy_pos = env.action_space.sample()
        while (deploy_pos == from_pos).all():
            deploy_pos = env.action_space.sample()
        if red_surrounded:
            env.step(deploy_pos)
        else:
            env.step(from_pos)
        if blue_surrounded:
            env.step(deploy_pos)
        else:
            env.step(from_pos)
    else:
        deploy_pos = (from_pos[0], from_pos[1] + 2)
        if env.width <= deploy_pos[1]:
            deploy_pos = (from_pos[0], from_pos[1] - 2)

    if red_surrounded:
        state, reward, terminated, truncated, info = env.step(from_pos)
    else:
        state, reward, terminated, truncated, info = env.step(deploy_pos)
    if blue_surrounded:
        state, reward, terminated, truncated, info = env.step(from_pos)
    else:
        state, reward, terminated, truncated, info = env.step(deploy_pos)

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
