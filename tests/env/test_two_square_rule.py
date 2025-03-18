
import itertools
import pytest

from stratego_gym.envs.primitives import Piece, Player
from stratego_gym.envs.startego import GamePhase

SCOUT_ONLY = {
    Piece.SCOUT: 1,
}

SPY_ONLY = {
    Piece.SPY: 1,
}

@pytest.mark.parametrize(
    "pices,direction", 
    itertools.product(
        [SCOUT_ONLY, SPY_ONLY],
        [True, False],
    )
)
def test_simple(env_5x5, pices, direction):
    env = env_5x5(pices)
    from_pos = (4, 2)
    to_pos = (3, 2) if direction else (4, 3)
    def move_fwd():
        env.step(from_pos)
        env.step(to_pos)
    def move_bwd():
        env.step(to_pos)
        env.step(from_pos)
    def repeat_twice(fn):
        fn()
        fn()
    env.step(from_pos)
    env.step(from_pos)
    assert env.game_phase == GamePhase.SELECT
    repeat_twice(move_fwd)
    repeat_twice(move_bwd)
    repeat_twice(move_fwd)

    piece = Piece(env.board[to_pos])
    assert not env.two_square_detector.validate_move(Player.RED, piece, to_pos, from_pos)
    assert not env.two_square_detector.validate_move(Player.BLUE, piece, to_pos, from_pos)

    move_bwd()
    assert len(env.two_square_detector.p1) == 1
    assert Piece(env.board[from_pos]) == Piece.EMPTY
    
    move_bwd()
    assert len(env.two_square_detector.p2) == 1
    assert Piece(env.board[from_pos]) == Piece.EMPTY
