
import itertools
import pytest

import numpy as np

from stratego_gym.envs.primitives import Piece, Player
from stratego_gym.envs.startego import GamePhase

SCOUT_ONLY = {
    Piece.SCOUT: 1,
}

SPY_ONLY = {
    Piece.SPY: 1,
}

SCOUT_PAIR = {
    Piece.SCOUT: 2,
}

SPY_PAIR = {
    Piece.SPY: 2,
}

def move_fwd(env, from_pos, to_pos):
    env.step(from_pos)
    env.step(to_pos)

def move_bwd(env, from_pos, to_pos):
    env.step(to_pos)
    env.step(from_pos)

def repeat_twice(fn, *args, **kwargs):
    fn(*args, **kwargs)
    fn(*args, **kwargs)

def rotate_pos(pos, height, width):
    return (height - pos[0] - 1, width - pos[1] - 1)

def validate_move(env, piece, from_pos, to_pos, maybe_occupied=False):
    from_pos_opponent = rotate_pos(from_pos, env.height, env.width)
    to_pos_opponent = rotate_pos(to_pos, env.height, env.width)
    for fp in (from_pos, from_pos_opponent):
        assert Piece(env.board[fp]) == Piece.EMPTY
    try:
        assert Piece(env.board[to_pos]) == piece
    except ValueError as e:
        if not maybe_occupied:
            raise e
    try:
        assert Piece(-env.board[to_pos_opponent]) == piece
    except ValueError as e:
        if not maybe_occupied:
            raise e

@pytest.mark.parametrize(
    "pices,from_pos", 
    itertools.product(
        [SCOUT_ONLY, SPY_ONLY],
        itertools.product(range(3, 5), range(5)),
    )
)
def test_one_step_move(env_5x5, pices, from_pos):
    env = env_5x5(pices)
    move_direction1 = np.array([-1, 1, 0, 0])
    move_direction2 = np.roll(move_direction1, 2)
    for direction in zip(move_direction1, move_direction2):    
        to_pos = (from_pos[0] + direction[0], from_pos[1] + direction[1])
        if to_pos[0] < 0 or env.height <= to_pos[0] or \
           to_pos[1] < 0 or env.width <= to_pos[1] or \
           env.lakes[to_pos]:
           continue

        env.reset()
        repeat_twice(env.step, from_pos)
        piece = Piece(env.board[from_pos])
        assert piece.value > Piece.LAKE.value
        assert env.game_phase == GamePhase.SELECT
        repeat_twice(move_fwd, env, from_pos, to_pos)
        validate_move(env, piece, from_pos, to_pos)

        repeat_twice(move_bwd, env, from_pos, to_pos)
        validate_move(env, piece, to_pos, from_pos)

        repeat_twice(move_fwd, env, from_pos, to_pos)
        validate_move(env, piece, from_pos, to_pos)

        assert not env.two_square_detector.validate_move(Player.RED, piece, to_pos, from_pos)
        assert not env.two_square_detector.validate_move(Player.BLUE, piece, to_pos, from_pos)
        assert len(env.two_square_detector.p1) == 3
        assert len(env.two_square_detector.p2) == 3

        repeat_twice(move_bwd, env, from_pos, to_pos)
        validate_move(env, Piece.EMPTY, to_pos, from_pos)
        assert len(env.two_square_detector.p1) == 1
        assert len(env.two_square_detector.p2) == 1


@pytest.mark.parametrize(
    "pices,from_pos", 
    itertools.product(
        [SCOUT_ONLY, SPY_ONLY],
        itertools.product(range(3, 5), range(5)),
    )
)
def test_no_violation_three_steps(env_5x5, pices, from_pos):
    env = env_5x5(pices)
    move_direction1 = np.array([-1, 1, 0, 0])
    move_direction2 = np.roll(move_direction1, 2)
    move_direction3 = np.roll(move_direction1, 1)
    move_direction4 = np.roll(move_direction1, 3)
    for direction1, direction2 in zip(zip(move_direction1, move_direction2), 
                                      zip(move_direction3, move_direction4)):    
        to_pos1 = (from_pos[0] + direction1[0], from_pos[1] + direction1[1])
        for i in range(2):
            if not i:
                # opposite direction
                to_pos2 = (from_pos[0] - direction1[0], from_pos[1] - direction1[1])
            else:
                # corner movement
                to_pos2 = (from_pos[0] + direction2[0], from_pos[1] + direction2[1])
            
            if to_pos1[0] < 0 or env.height <= to_pos1[0]  or \
               to_pos1[1] < 0 or env.width <= to_pos1[1]  or \
               env.lakes[to_pos1]:
                continue
            if to_pos2[0] < 0 or env.height <= to_pos2[0]  or \
               to_pos2[1] < 0 or env.width <= to_pos2[1]  or \
               env.lakes[to_pos2]:
                continue
            
            env.reset()
            repeat_twice(env.step, from_pos)
            piece = Piece(env.board[from_pos])
            assert piece.value > Piece.LAKE.value
            assert env.game_phase == GamePhase.SELECT
            for to_pos in 2 * [to_pos1, to_pos2]:
                repeat_twice(move_fwd, env, from_pos, to_pos)
                validate_move(env, piece, from_pos, to_pos)

                repeat_twice(move_bwd, env, from_pos, to_pos)
                validate_move(env, piece, to_pos, from_pos)
                assert len(env.two_square_detector.p1) == 2
                assert len(env.two_square_detector.p2) == 2


@pytest.mark.parametrize(
    "pices,from_pos", 
    itertools.product(
        [SCOUT_ONLY, SPY_ONLY],
        itertools.product(range(3, 5), range(5)),
    )
)
def test_no_violation_square_movement(env_5x5, pices, from_pos):
    env = env_5x5(pices)
    move_direction1 = np.array(2 * [-1, 0, 1, 0])
    move_direction2 = np.roll(move_direction1, 1)
    for clockwise in range(2):
        for _ in range(4):
            env.reset()
            repeat_twice(env.step, from_pos)
            piece = Piece(env.board[from_pos])
            assert piece.value > Piece.LAKE.value
            assert env.game_phase == GamePhase.SELECT
            _from_pos = from_pos
            for direction in zip(move_direction1, move_direction2):
                to_pos = (_from_pos[0] + direction[0], _from_pos[1] + direction[1])
                if to_pos[0] < 0 or env.height <= to_pos[0] or \
                   to_pos[1] < 0 or env.width <= to_pos[1] or \
                   env.lakes[to_pos]:
                    break
                repeat_twice(move_fwd, env, _from_pos, to_pos)
                validate_move(env, piece, _from_pos, to_pos)
                assert len(env.two_square_detector.p1) == 1
                assert len(env.two_square_detector.p2) == 1
                _from_pos = to_pos
            np.roll(move_direction1, 1)
            np.roll(move_direction2, 1)
        move_direction1, move_direction2 = move_direction2, move_direction1


@pytest.mark.parametrize(
    "pices,from_pos", 
    itertools.product(
        [SCOUT_PAIR, SPY_PAIR],
        itertools.product(range(3, 5), range(1, 4)),
    )
)
def test_no_violation_interrupt(env_5x5, pices, from_pos):
    env = env_5x5(pices)
    move_direction1 = np.array([0, 0])
    move_direction2 = np.array([-1, 1])
    for direction in zip(move_direction1, move_direction2):    
        to_pos = (from_pos[0] + direction[0], from_pos[1] + direction[1])
        if to_pos[0] < 0 or env.height <= to_pos[0] or \
           to_pos[1] < 0 or env.width <= to_pos[1] or \
           env.lakes[to_pos]:
           continue

        from_pos2 = (3 + 4 - from_pos[0], from_pos[1])
        to_pos2 = (from_pos2[0] - direction[0], from_pos2[1] - direction[1])
        if from_pos2[0] < 0 or env.height <= from_pos2[0] or \
           from_pos2[1] < 0 or env.width <= from_pos2[1] or \
           env.lakes[from_pos2]:
           continue
        if to_pos2[0] < 0 or env.height <= to_pos2[0] or \
           to_pos2[1] < 0 or env.width <= to_pos2[1] or \
           env.lakes[to_pos2]:
           continue

        env.reset()
        repeat_twice(env.step, from_pos)
        repeat_twice(env.step, from_pos2)
        piece = Piece(env.board[from_pos])
        assert piece.value > Piece.LAKE.value
        assert env.game_phase == GamePhase.SELECT
        repeat_twice(move_fwd, env, from_pos, to_pos)
        validate_move(env, piece, from_pos, to_pos)

        repeat_twice(move_bwd, env, from_pos, to_pos)
        validate_move(env, piece, to_pos, from_pos)

        repeat_twice(move_fwd, env, from_pos, to_pos)
        validate_move(env, piece, from_pos, to_pos)

        repeat_twice(move_fwd, env, from_pos2, to_pos2)
        validate_move(env, piece, from_pos2, to_pos2)

        assert env.two_square_detector.validate_move(Player.RED, piece, to_pos, from_pos)
        assert env.two_square_detector.validate_move(Player.BLUE, piece, to_pos, from_pos)
        assert len(env.two_square_detector.p1) == 1
        assert len(env.two_square_detector.p2) == 1

        repeat_twice(move_bwd, env, from_pos, to_pos)
        validate_move(env, piece, to_pos, from_pos)
        assert len(env.two_square_detector.p1) == 1
        assert len(env.two_square_detector.p2) == 1


@pytest.mark.parametrize(
    "pices,from_pos", 
    itertools.product(
        [SCOUT_PAIR, SPY_PAIR],
        itertools.product(range(3, 5), range(1, 4)),
    )
)
def test_no_violation_interrupt_sync(env_5x5, pices, from_pos):
    env = env_5x5(pices)
    move_direction1 = np.array([-1, 1, 0, 0])
    move_direction2 = np.roll(move_direction1, 2)
    for direction in zip(move_direction1, move_direction2):    
        to_pos = (from_pos[0] + direction[0], from_pos[1] + direction[1])
        if to_pos[0] < 0 or env.height <= to_pos[0] or \
           to_pos[1] < 0 or env.width <= to_pos[1] or \
           env.lakes[to_pos]:
           continue

        from_pos2 = (from_pos[0], from_pos[1] + 1)
        to_pos2 = (from_pos2[0] + direction[0], from_pos2[1] + direction[1])
        if from_pos2[0] < 0 or env.height <= from_pos2[0] or \
           from_pos2[1] < 0 or env.width <= from_pos2[1] or \
           from_pos2 == to_pos or env.lakes[from_pos2]:
           continue

        if to_pos2[0] < 0 or env.height <= to_pos2[0] or \
           to_pos2[1] < 0 or env.width <= to_pos2[1] or \
           from_pos == to_pos2 or env.lakes[to_pos2]:
           continue

        env.reset()
        repeat_twice(env.step, from_pos)
        repeat_twice(env.step, from_pos2)
        piece = Piece(env.board[from_pos])
        assert piece.value > Piece.LAKE.value
        assert env.game_phase == GamePhase.SELECT
        
        for fp, tp in zip((from_pos, from_pos2), (to_pos, to_pos2)):
            repeat_twice(move_fwd, env, fp, tp)
            validate_move(env, piece, fp, tp)

            assert len(env.two_square_detector.p1) == 1
            assert len(env.two_square_detector.p2) == 1

        for fp, tp in zip((from_pos, from_pos2), (to_pos, to_pos2)):
            repeat_twice(move_bwd, env, fp, tp)
            validate_move(env, piece, tp, fp)

            assert len(env.two_square_detector.p1) == 1
            assert len(env.two_square_detector.p2) == 1


@pytest.mark.parametrize(
    "pices,from_pos", 
    itertools.product(
        [SCOUT_ONLY,],
        itertools.product(range(3, 5), range(5)),
    )
)
def test_scout_move_entire_row(env_5x5, pices, from_pos):
    env = env_5x5(pices)
    move_direction1 = np.array([-1, 1, 0, 0])
    move_direction2 = np.roll(move_direction1, 2)
    for direction in zip(move_direction1, move_direction2):
        to_pos = from_pos
        while True:    
            _to_pos = (to_pos[0] + direction[0], to_pos[1] + direction[1])
            if _to_pos[0] < 0 or env.height <= _to_pos[0] or \
               _to_pos[1] < 0 or env.width <= _to_pos[1] or \
               env.lakes[_to_pos]:
                break
            to_pos = _to_pos
        if to_pos == from_pos:
            continue

        env.reset()
        repeat_twice(env.step, from_pos)
        piece = Piece(env.board[from_pos])
        assert piece.value > Piece.LAKE.value
        assert env.game_phase == GamePhase.SELECT
        repeat_twice(move_fwd, env, from_pos, to_pos)
        validate_move(env, piece, from_pos, to_pos)

        repeat_twice(move_bwd, env, from_pos, to_pos)
        validate_move(env, piece, to_pos, from_pos)

        repeat_twice(move_fwd, env, from_pos, to_pos)
        validate_move(env, piece, from_pos, to_pos)

        assert not env.two_square_detector.validate_move(Player.RED, piece, to_pos, from_pos)
        assert not env.two_square_detector.validate_move(Player.BLUE, piece, to_pos, from_pos)
        assert len(env.two_square_detector.p1) == 3
        assert len(env.two_square_detector.p2) == 3

        repeat_twice(move_bwd, env, from_pos, to_pos)
        validate_move(env, Piece.EMPTY, to_pos, from_pos, maybe_occupied=True)
        assert len(env.two_square_detector.p1) == 1
        assert len(env.two_square_detector.p2) == 1
