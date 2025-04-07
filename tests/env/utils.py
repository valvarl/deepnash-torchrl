from stratego.core.primitives import Piece, Pos
from stratego.core.stratego import StrategoEnv
from stratego.core.stratego_base import StrategoEnvBase
from stratego.wrappers.cpp_env import StrategoEnvCpp


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

FLAG_2BOMB_SCOUT = {
    Piece.FLAG: 1,
    Piece.BOMB: 2,
    Piece.SPY: 1,
}

FLAG_2BOMB_SPY = {
    Piece.FLAG: 1,
    Piece.BOMB: 2,
    Piece.SPY: 1,
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


def validate_move(
    env: StrategoEnv,
    piece: Piece,
    from_pos: Pos,
    to_pos: Pos,
    maybe_occupied: bool = False,
):
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


class StrategoEnvIdentityChecker(StrategoEnvBase):
    def __init__(
        self, py_env: StrategoEnv, cpp_env: StrategoEnvCpp, return_compile=False
    ):
        self.py_env = py_env
        self.cpp_env = cpp_env
        self.return_compile = return_compile

    def reset(self, seed=None, options=None):
        py_out = self.py_env.reset(seed, options)
        cpp_out = self.cpp_env.reset(seed, options)
        valid, msg = self.validate_outputs(py_out, cpp_out)
        assert valid, msg
        valid, msg = self.validate_states()
        assert valid, msg
        return py_out if not self.return_compile else cpp_out

    def step(self, action):
        py_out = self.py_env.step(action)
        cpp_out = self.cpp_env.step(action)
        valid, msg = self.validate_outputs(py_out, cpp_out)
        assert valid, msg
        valid, msg = self.validate_states()
        assert valid, msg
        return py_out if not self.return_compile else cpp_out

    def __getattr__(self, name):
        return getattr(self.py_env, name)

    def validate_outputs(self, py_out, cpp_out) -> tuple[bool, str]:
        py_obs, py_action_mask, py_reward, py_treminated, py_truncated, py_info = py_out
        cpp_obs, cpp_action_mask, cpp_reward, cpp_terminated, cpp_truncated = cpp_out

        if (py_obs != cpp_obs).any():
            return False, "obs"

        if (py_action_mask != cpp_action_mask).any():
            return False, "action_mask"

        if py_reward != cpp_reward:
            return False, "reward"

        if py_treminated != cpp_terminated:
            return False, "treminated"

        if py_truncated != cpp_truncated:
            return False, "truncated"

        # TODO: complete info check

        return True, "Outputs match"

    def validate_states(self):
        if self.py_env.height != self.cpp_env.height:
            return False, "height"

        if self.py_env.width != self.cpp_env.width:
            return False, "width"

        if self.py_env.player != self.cpp_env.current_player:
            return False, "player"

        if self.py_env.game_phase != self.cpp_env.game_phase:
            return False, "game_phase"

        if (self.py_env.board != self.cpp_env.board).any():
            return False, "board"

        if (self.py_env.lakes != self.cpp_env.lakes).any():
            return False, "lakes"

        # TODO: complete checks

        return True, "States match"
