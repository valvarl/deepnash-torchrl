from stratego_gym.envs.primitives import Piece, Pos
from stratego_gym.envs.startego import StrategoEnv


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

def move_fwd(env: StrategoEnv, from_pos, to_pos):
    env.step(from_pos)
    env.step(to_pos)

def move_bwd(env: StrategoEnv, from_pos, to_pos):
    env.step(to_pos)
    env.step(from_pos)

def repeat_twice(fn, *args, **kwargs):
    fn(*args, **kwargs)
    fn(*args, **kwargs)

def rotate_pos(pos, height, width):
    return (height - pos[0] - 1, width - pos[1] - 1)

def validate_move(env: StrategoEnv, piece: Piece, from_pos: Pos, to_pos: Pos, maybe_occupied: bool = False):
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
