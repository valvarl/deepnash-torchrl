import itertools
from typing import Any, Callable

import numpy as np
import pytest

from stratego_gym.envs.primitives import Piece, Player, Pos
from stratego_gym.envs.startego import GamePhase, StrategoEnv
from tests.env.utils import (
    FLAG_2BOMB_SCOUT, FLAG_2BOMB_SPY,
    SCOUT_ONLY, SPY_ONLY, SCOUT_PAIR, SPY_PAIR,
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

@pytest.mark.parametrize(
    "pieces,red_stucked,blue_stucked",
    itertools.product(
        [SCOUT_ONLY, SPY_ONLY],
        [True, False],
        [True, False],
    )
)
def test_two_square_rule_piece_stucked(env_5x5, pieces, red_stucked, blue_stucked):
    """
    ...L.
    ....B
    LLLLL <- lakes are here so the scout can't escape
    R....
    .L...
    """
    lakes_mask = np.zeros((5, 5), dtype=bool)
    lakes_mask[4, 1] = True
    lakes_mask[0, 3] = True
    lakes_mask[2] = True
    p1_deploy_mask = np.zeros((5, 5), dtype=bool)
    p2_deploy_mask = np.zeros((5, 5), dtype=bool)
    p1_deploy_mask[3:] = True
    p2_deploy_mask[:2] = True
    p1_deploy_mask &= ~lakes_mask
    p2_deploy_mask &= ~lakes_mask

    env = env_5x5(pieces, lakes_mask=lakes_mask, p1_deploy_mask=p1_deploy_mask, p2_deploy_mask=p2_deploy_mask)
    from_pos = (3, 0)
    repeat_twice(env.step, from_pos)
    piece = Piece(env.board[from_pos])
    assert piece.value > Piece.LAKE.value
    assert env.game_phase == GamePhase.SELECT
    stuck_direction = np.array([1, -1, 1])
    from_pos_red = from_pos
    from_pos_blue = from_pos
    for i, direction in enumerate(stuck_direction):
        if red_stucked:
            to_pos_red = (from_pos_red[0] + direction, from_pos_red[1])
        else:
            to_pos_red = (from_pos_red[0], from_pos_red[1] + 1)
        state, reward, terminated, truncated, info = env.step(from_pos_red)
        state, reward, terminated, truncated, info = env.step(to_pos_red)
        from_pos_red = to_pos_red

        if blue_stucked:
            to_pos_blue = (from_pos_blue[0] + direction, from_pos_blue[1])
        else:
            to_pos_blue = (from_pos_blue[0], from_pos_blue[1] + 1)
        state, reward, terminated, truncated, info = env.step(from_pos_blue)
        state, reward, terminated, truncated, info = env.step(to_pos_blue)
        from_pos_blue = to_pos_blue

    if red_stucked and blue_stucked or red_stucked:
        assert env.game_phase == GamePhase.TERMINAL
        # If the game ends with blue's turn, but he can move, he is awarded 1 on his turn.
        assert (reward == 0) if blue_stucked else (reward == 1)
    elif blue_stucked:
        assert env.game_phase == GamePhase.SELECT
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.game_phase == GamePhase.TERMINAL
        assert reward == 1
    else:
        # not red_stucked and not blue_stucked
        pass

@pytest.mark.parametrize(
    "pieces,red_stucked,blue_stucked",
    itertools.product(
        [SCOUT_PAIR, SPY_PAIR],
        [True, False],
        [True, False],
    )
)
def test_two_square_rule_two_pieces_stucked(env_5x5, pieces, red_stucked, blue_stucked):
    """
    B.sL.
    .bb..
    .....
    ..bb.
    .Ls.R
    """
    lakes_mask = np.zeros((5, 5), dtype=bool)
    lakes_mask[4, 1] = True
    lakes_mask[0, 3] = True
    p1_deploy_mask = np.zeros((5, 5), dtype=bool)
    p2_deploy_mask = np.zeros((5, 5), dtype=bool)
    p1_deploy_mask[3:] = True
    p2_deploy_mask[:2] = True
    p1_deploy_mask &= ~lakes_mask
    p2_deploy_mask &= ~lakes_mask

    _pieces = {Piece.BOMB: 2}
    _pieces.update(pieces)
    env = env_5x5(_pieces, lakes_mask=lakes_mask, p1_deploy_mask=p1_deploy_mask, p2_deploy_mask=p2_deploy_mask)
    deploy_pos = [(3, 2), (3, 3), (4, 2), (4, 4)]
    for from_pos in deploy_pos:
        repeat_twice(env.step, from_pos)

    assert env.game_phase == GamePhase.SELECT
    stuck_direction = np.array([-1, 1, -1])
    from_pos_red = from_pos
    from_pos_blue = from_pos
    for i, direction in enumerate(stuck_direction):
        if red_stucked:
            to_pos_red = (from_pos_red[0], from_pos_red[1] + direction)
        else:
            to_pos_red = (from_pos_red[0] - 1, from_pos_red[1])
        state, reward, terminated, truncated, info = env.step(from_pos_red)
        state, reward, terminated, truncated, info = env.step(to_pos_red)
        from_pos_red = to_pos_red

        if blue_stucked:
            to_pos_blue = (from_pos_blue[0], from_pos_blue[1] + direction)
        else:
            to_pos_blue = (from_pos_blue[0] - 1, from_pos_blue[1])
        state, reward, terminated, truncated, info = env.step(from_pos_blue)
        state, reward, terminated, truncated, info = env.step(to_pos_blue)
        from_pos_blue = to_pos_blue

    if red_stucked and blue_stucked or red_stucked:
        assert env.game_phase == GamePhase.TERMINAL
        # If the game ends with blue's turn, but he can move, he is awarded 1 on his turn.
        assert (reward == 0) if blue_stucked else (reward == 1)
    elif blue_stucked:
        assert env.game_phase == GamePhase.SELECT
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.game_phase == GamePhase.TERMINAL
        assert reward == 1
    else:
        # not red_stucked and not blue_stucked
        pass

@pytest.mark.parametrize(
    "pieces,red_no_pieces,blue_no_pieces",
    itertools.product(
        [SCOUT_PAIR, SPY_PAIR],
        [True, False],
        [True, False],
    )
)
def test_no_pieces_left(env_5x5, pieces, red_no_pieces, blue_no_pieces):
    """
    sb..B
    b....
    ..L..
    ....b
    R..bs
    """
    _pieces = {Piece.BOMB: 2}
    _pieces.update(pieces)
    env = env_5x5(_pieces)
    deploy_pos = [(3, 4), (4, 3), (4, 4)]
    for from_pos in deploy_pos:
        repeat_twice(env.step, from_pos)

    from_pos_red = (4, 0)
    if not red_no_pieces:
        from_pos_red = (4, 1)
    from_pos_blue = (4, 0)
    if not blue_no_pieces:
        from_pos_blue = (4, 1)

    env.step(from_pos_red)
    env.step(from_pos_blue)

    assert env.game_phase == GamePhase.SELECT
    for i in range(3):
        to_pos_red = (from_pos_red[0] - 1, from_pos_red[1])
        state, reward, terminated, truncated, info = env.step(from_pos_red)
        state, reward, terminated, truncated, info = env.step(to_pos_red)
        from_pos_red = to_pos_red

        to_pos_blue = (from_pos_blue[0] - 1, from_pos_blue[1])
        state, reward, terminated, truncated, info = env.step(from_pos_blue)
        state, reward, terminated, truncated, info = env.step(to_pos_blue)
        from_pos_blue = to_pos_blue

    if red_no_pieces and blue_no_pieces or red_no_pieces:
        assert env.game_phase == GamePhase.TERMINAL
        # If the game ends with blue's turn, but he can move, he is awarded 1 on his turn.
        assert (reward == 0) if blue_no_pieces else (reward == 1)
    elif blue_no_pieces:
        assert env.game_phase == GamePhase.SELECT
        to_pos_red = (from_pos_red[0], from_pos_red[1] + 1)
        state, reward, terminated, truncated, info = env.step(from_pos_red)
        state, reward, terminated, truncated, info = env.step(to_pos_red)
        assert env.game_phase == GamePhase.TERMINAL
        assert reward == 1
    else:
        # not red_no_pieces and not blue_no_pieces
        pass

@pytest.mark.parametrize(
    "pieces,red_attacker",
    itertools.product(
        [SCOUT_PAIR, SPY_PAIR],
        [True, False],
    )
)
def test_no_pieces_left_on_trade(env_5x5, pieces, red_attacker):
    """
    sbB..
    b....
    .....
    ....b
    ..Rbs
    """
    _pieces = {Piece.BOMB: 2}
    _pieces.update(pieces)
    env = env_5x5(_pieces, lakes=[])
    deploy_pos = [(3, 4), (4, 3), (4, 4)]
    for from_pos in deploy_pos:
        repeat_twice(env.step, from_pos)

    from_pos_red = (3, 2)
    if not red_attacker:
        from_pos_red = (4, 2)
    from_pos_blue = (4, 2)

    env.step(from_pos_red)
    env.step(from_pos_blue)

    assert env.game_phase == GamePhase.SELECT
    for i in range(2):
        to_pos_red = (from_pos_red[0] - 1, from_pos_red[1])
        state, reward, terminated, truncated, info = env.step(from_pos_red)
        state, reward, terminated, truncated, info = env.step(to_pos_red)
        from_pos_red = to_pos_red

        if i == 1 and red_attacker:
            break

        to_pos_blue = (from_pos_blue[0] - 1, from_pos_blue[1])
        state, reward, terminated, truncated, info = env.step(from_pos_blue)
        state, reward, terminated, truncated, info = env.step(to_pos_blue)
        from_pos_blue = to_pos_blue

    assert env.game_phase == GamePhase.TERMINAL
    assert reward == 0

@pytest.mark.parametrize(
    "pieces,red_attacker",
    itertools.product(
        [FLAG_2BOMB_SCOUT, FLAG_2BOMB_SPY],
        [True, False],
    )
)
def test_flag_captured(env_5x5, pieces, red_attacker):
    """
    bb..B
    f....
    ..L..
    ....f
    R..bb
    """
    env = env_5x5(pieces)
    deploy_pos = [(3, 4), (4, 3), (4, 4)]
    for from_pos in deploy_pos:
        repeat_twice(env.step, from_pos)

    from_pos_red = (4, 0)
    from_pos_blue = (4, 1)
    if not red_attacker:
        from_pos_red = (4, 1)
        from_pos_blue = (4, 0)

    env.step(from_pos_red)
    env.step(from_pos_blue)

    assert env.game_phase == GamePhase.SELECT
    for i in range(3):
        to_pos_red = (from_pos_red[0] - 1, from_pos_red[1])
        state, reward, terminated, truncated, info = env.step(from_pos_red)
        state, reward, terminated, truncated, info = env.step(to_pos_red)
        from_pos_red = to_pos_red

        if i == 2 and red_attacker:
            break

        to_pos_blue = (from_pos_blue[0] - 1, from_pos_blue[1])
        state, reward, terminated, truncated, info = env.step(from_pos_blue)
        state, reward, terminated, truncated, info = env.step(to_pos_blue)
        from_pos_blue = to_pos_blue

    assert env.game_phase == GamePhase.TERMINAL
    assert reward == 1

@pytest.mark.parametrize(
    "pieces,total_moves_limit",
    itertools.product(
        [SCOUT_ONLY, SPY_ONLY],
        range(1, 11),
    )
)
def test_total_moves_limit(env_5x5, pieces, total_moves_limit):
    lakes_mask = np.zeros((5, 5), dtype=bool)
    lakes_mask[2] = True
    p1_deploy_mask = np.zeros((5, 5), dtype=bool)
    p2_deploy_mask = np.zeros((5, 5), dtype=bool)
    p1_deploy_mask[3:] = True
    p2_deploy_mask[:2] = True
    p1_deploy_mask &= ~lakes_mask
    p2_deploy_mask &= ~lakes_mask

    env = env_5x5(pieces, lakes_mask=lakes_mask, p1_deploy_mask=p1_deploy_mask, p2_deploy_mask=p2_deploy_mask)
    env.total_moves_limit = total_moves_limit

    repeat_twice(env.step, env.action_space.sample())

    for i in range(total_moves_limit + 1):
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())

    assert env.game_phase == GamePhase.TERMINAL
    assert reward == 0

@pytest.mark.parametrize(
    "pieces,moves_since_attack_limit",
    itertools.product(
        [SCOUT_PAIR, SPY_PAIR],
        range(1, 11),
    )
)
def test_moves_since_attack_limit(env_5x5, pieces, moves_since_attack_limit):
    """
    B.L.s
    ..LLb
    LLLLL
    bLL..
    s.L.R
    """
    lakes_mask = np.zeros((5, 5), dtype=bool)
    lakes_mask[2] = True
    lakes_mask[:2, 2] = True
    lakes_mask[1, 3] = True
    lakes_mask[3, 1] = True
    lakes_mask[3:, 2] = True
    p1_deploy_mask = np.zeros((5, 5), dtype=bool)
    p2_deploy_mask = np.zeros((5, 5), dtype=bool)
    p1_deploy_mask[3:] = True
    p2_deploy_mask[:2] = True
    p1_deploy_mask[1, 4] = True
    p2_deploy_mask[3, 0] = True
    p2_deploy_mask[1, 4] = False
    p1_deploy_mask[3, 0] = False
    p1_deploy_mask &= ~lakes_mask
    p2_deploy_mask &= ~lakes_mask

    _pieces = {Piece.BOMB: 1}
    _pieces.update(pieces)
    env = env_5x5(_pieces, lakes_mask=lakes_mask, p1_deploy_mask=p1_deploy_mask, p2_deploy_mask=p2_deploy_mask)
    env.moves_since_attack_limit = moves_since_attack_limit
    
    deploy_pos = [(1, 4), (4, 0), (4, 4)]
    for from_pos in deploy_pos:
        repeat_twice(env.step, from_pos)

    assert env.game_phase == GamePhase.SELECT
    moves = [((4, 0), (4, 1)), ((4, 1), (4, 0)), ((4, 0), (3, 0))]
    for i, move in enumerate(moves):
        assert env.draw_conditions["moves_since_attack"] == 2 * i
        move_fwd(env, *move)
        if env.game_phase == GamePhase.TERMINAL:
            return
        assert env.draw_conditions["moves_since_attack"] == 2 * i + 1 or i == 2
        move_fwd(env, *move)
        if env.game_phase == GamePhase.TERMINAL:
            return

    assert env.draw_conditions["moves_since_attack"] == 0
    assert env.game_phase == GamePhase.SELECT
    for i in range(moves_since_attack_limit + 1):
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())

    assert env.game_phase == GamePhase.TERMINAL
    assert env.draw_conditions["total_moves"] == 6 + moves_since_attack_limit
    assert reward == 0
