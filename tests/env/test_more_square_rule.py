import itertools
from typing import Any, Callable

import numpy as np
import pytest

from stratego_gym.envs.primitives import Piece, Player, Pos
from stratego_gym.envs.startego import GamePhase, StrategoEnv
from tests.env.utils import SCOUT_ONLY, SPY_ONLY, move_fwd, repeat_twice, rotate_pos, validate_move


@pytest.mark.parametrize(
    "clockwise_first", 
    itertools.product(
        [True, False],
    )
)
def test_scout_circle_chasing(env_5x5: Callable[[Any], StrategoEnv], clockwise_first: bool):
    """Position before the start of chasing:
    ....B
    .....
    ..L..
    .....
    R....
    
    Chase initiation position:
    R...B
    .....
    ..L..
    .....
    .....
    """
    env = env_5x5(SCOUT_ONLY)
    from_pos = (4, 2)
    pos_list = [(4, 0), (0, 0), (0, 4), (4, 4), (4, 0)]
    if not clockwise_first:
        pos_list[1], pos_list[3] = pos_list[3], pos_list[1]
    chase_init_pos = pos_list[1]
    repeat_twice(env.step, from_pos)
    piece = Piece(env.board[from_pos])
    assert piece.value > Piece.LAKE.value
    assert env.game_phase == GamePhase.SELECT
    for to_pos in pos_list:
        repeat_twice(move_fwd, env, from_pos, to_pos)
        validate_move(env, piece, from_pos, to_pos)
        from_pos = to_pos
    assert not env.chasing_detector.validate_move(Player.RED, piece, from_pos, chase_init_pos, env.board)
    
    # counter-clockwise pursuit start
    chase_init_cc_pos = (4, 4)
    for to_pos in pos_list[::-1][1:]:
        repeat_twice(move_fwd, env, from_pos, to_pos)
        validate_move(env, piece, from_pos, to_pos)
        from_pos = to_pos
    assert not env.chasing_detector.validate_move(Player.RED, piece, from_pos, chase_init_pos, env.board)
    assert not env.chasing_detector.validate_move(Player.RED, piece, from_pos, chase_init_cc_pos, env.board)
    

def test_scout_circle_chasing_two_step(env_5x5: Callable[[Any], StrategoEnv]):
    """A chase involving the midpoints of the board sides.
    Since Blue does not get rid of the pursuit, but continues on the edge, 
    at this point he becomes the attacker, interrupting the sequence.
    
    Chase initiation position:
    ....R
    .....
    ..L.B
    .....
    ..... <- corner move interrupts the chase
    """
    env = env_5x5(SCOUT_ONLY)
    from_pos = (4, 2)
    pos_list = [(0, 0), (0, 2), (0, 4), (2, 4), (4, 4), (4, 2), (4, 0), (2, 0), (0, 0)]
    repeat_twice(env.step, from_pos)
    piece = Piece(env.board[from_pos])
    assert piece.value > Piece.LAKE.value
    assert env.game_phase == GamePhase.SELECT

    repeat_twice(move_fwd, env, from_pos, (4, 0))
    validate_move(env, piece, from_pos, (4, 0))
    move_fwd(env, (4, 0), (0, 0))
    move_fwd(env, (4, 0), (2, 0))
    move_fwd(env, (0, 0), (0, 4))

    red_from_pos = (0, 4)
    blue_from_pos = (2, 0)
    for blue_to_pos in pos_list:
        move_fwd(env, blue_from_pos, blue_to_pos)
        move_fwd(env, red_from_pos, rotate_pos(blue_from_pos, env.height, env.width))
        if 2 not in blue_to_pos:
            assert len(env.chasing_detector.chase_moves) == 1
        red_from_pos = rotate_pos(blue_from_pos, env.height, env.width)
        blue_from_pos = blue_to_pos


@pytest.mark.parametrize(
    "pieces,red_from_col,red_attacker",
    itertools.product(
        [SCOUT_ONLY, SPY_ONLY],
        range(5),
        [True, False],
    )
)
def test_three_square_chasing_opposition(env_5x5: Callable[[Any], StrategoEnv], pieces, red_from_col, red_attacker: bool):
    env = env_5x5(pieces, lakes=[])
    
    move_direction = np.array([-1, -1, 1, 1])
    for _ in range(4):
        env.reset()
        red_from_pos = (4 if red_attacker else 3, red_from_col)
        blue_from_pos = (4, env.width - red_from_col - 1)
        env.step(red_from_pos)
        env.step(blue_from_pos)
        piece = Piece(env.board[red_from_pos])
        assert piece.value > Piece.LAKE.value
        assert env.game_phase == GamePhase.SELECT
        red_to_pos = (red_from_pos[0] - 1, red_from_pos[1])
        move_fwd(env, red_from_pos, red_to_pos)
        red_from_pos = red_to_pos
        blue_to_pos = (blue_from_pos[0] - 1, blue_from_pos[1])
        move_fwd(env, blue_from_pos, blue_to_pos)
        blue_from_pos = blue_to_pos

        if red_attacker:
            red_to_pos = (red_from_pos[0] - 1, red_from_pos[1])
            move_fwd(env, red_from_pos, red_to_pos)
            red_from_pos = red_to_pos

        for direction in move_direction:
            blue_to_pos = (blue_to_pos[0], blue_to_pos[1] + int(direction))
            if blue_to_pos[0] < 0 or env.height <= blue_to_pos[0] or \
                blue_to_pos[1] < 0 or env.width <= blue_to_pos[1] or \
                env.lakes[blue_to_pos]:
                break
            red_to_pos = (red_to_pos[0], red_to_pos[1] - int(direction))
            if red_to_pos[0] < 0 or env.height <= red_to_pos[0] or \
                red_to_pos[1] < 0 or env.width <= red_to_pos[1] or \
                env.lakes[red_to_pos]:
                break
            if red_attacker:
                move_fwd(env, blue_from_pos, blue_to_pos)
                blue_from_pos = blue_to_pos
                if len(env.chasing_detector.chase_moves) == 8:
                    assert not env.chasing_detector.validate_move(Player.RED, piece, red_from_pos, red_to_pos, env.board)
                move_fwd(env, red_from_pos, red_to_pos)
                red_from_pos = red_to_pos
            else:
                move_fwd(env, red_from_pos, red_to_pos)
                red_from_pos = red_to_pos
                if len(env.chasing_detector.chase_moves) == 8:
                    assert not env.chasing_detector.validate_move(Player.BLUE, piece, rotate_pos(blue_from_pos, 5, 5), rotate_pos(blue_to_pos, 5, 5), env.board)
                move_fwd(env, blue_from_pos, blue_to_pos)
                blue_from_pos = blue_to_pos

        move_direction = np.roll(move_direction, 1)
                
