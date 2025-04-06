import itertools
from typing import Any, Callable
import pytest

import numpy as np

from stratego.core.primitives import Piece, Player
from stratego.core.startego import GamePhase, StrategoEnv
from tests.env.utils import (
    SCOUT_ONLY,
    SCOUT_PAIR,
    SPY_ONLY,
    SPY_PAIR,
    move_bwd,
    move_fwd,
    repeat_twice,
    validate_move,
)


@pytest.mark.parametrize(
    "pices,from_pos",
    itertools.product(
        [SCOUT_ONLY, SPY_ONLY],
        itertools.product(range(3, 5), range(5)),
    ),
)
def test_one_step_move(env_5x5: Callable[[Any], StrategoEnv], pices, from_pos):
    env = env_5x5(pices)
    move_direction1 = np.array([-1, 1, 0, 0])
    move_direction2 = np.roll(move_direction1, 2)
    for direction in zip(move_direction1, move_direction2):
        to_pos = (from_pos[0] + direction[0], from_pos[1] + direction[1])
        if (
            to_pos[0] < 0
            or env.height <= to_pos[0]
            or to_pos[1] < 0
            or env.width <= to_pos[1]
            or env.lakes[to_pos]
        ):
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

        assert not env.two_square_detector.validate_move(
            Player.RED, piece, to_pos, from_pos
        )
        assert not env.two_square_detector.validate_move(
            Player.BLUE, piece, to_pos, from_pos
        )
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
    ),
)
def test_no_violation_three_steps(
    env_5x5: Callable[[Any], StrategoEnv], pices, from_pos
):
    env = env_5x5(pices)
    move_direction1 = np.array([-1, 1, 0, 0])
    move_direction2 = np.roll(move_direction1, 2)
    move_direction3 = np.roll(move_direction1, 1)
    move_direction4 = np.roll(move_direction1, 3)
    for direction1, direction2 in zip(
        zip(move_direction1, move_direction2), zip(move_direction3, move_direction4)
    ):
        to_pos1 = (from_pos[0] + direction1[0], from_pos[1] + direction1[1])
        for i in range(2):
            if not i:
                # opposite direction
                to_pos2 = (from_pos[0] - direction1[0], from_pos[1] - direction1[1])
            else:
                # corner movement
                to_pos2 = (from_pos[0] + direction2[0], from_pos[1] + direction2[1])

            if (
                to_pos1[0] < 0
                or env.height <= to_pos1[0]
                or to_pos1[1] < 0
                or env.width <= to_pos1[1]
                or env.lakes[to_pos1]
            ):
                continue
            if (
                to_pos2[0] < 0
                or env.height <= to_pos2[0]
                or to_pos2[1] < 0
                or env.width <= to_pos2[1]
                or env.lakes[to_pos2]
            ):
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
    ),
)
def test_no_violation_square_movement(
    env_5x5: Callable[[Any], StrategoEnv], pices, from_pos
):
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
                if (
                    to_pos[0] < 0
                    or env.height <= to_pos[0]
                    or to_pos[1] < 0
                    or env.width <= to_pos[1]
                    or env.lakes[to_pos]
                ):
                    break
                repeat_twice(move_fwd, env, _from_pos, to_pos)
                validate_move(env, piece, _from_pos, to_pos)
                assert len(env.two_square_detector.p1) == 1
                assert len(env.two_square_detector.p2) == 1
                _from_pos = to_pos
            move_direction1 = np.roll(move_direction1, 1)
            move_direction2 = np.roll(move_direction2, 1)
        move_direction1, move_direction2 = move_direction2, move_direction1


@pytest.mark.parametrize(
    "pices,from_pos",
    itertools.product(
        [SCOUT_PAIR, SPY_PAIR],
        itertools.product(range(3, 5), range(1, 4)),
    ),
)
def test_no_violation_interrupt(env_5x5: Callable[[Any], StrategoEnv], pices, from_pos):
    env = env_5x5(pices)
    move_direction1 = np.array([0, 0])
    move_direction2 = np.array([-1, 1])
    for direction in zip(move_direction1, move_direction2):
        to_pos = (from_pos[0] + direction[0], from_pos[1] + direction[1])
        if (
            to_pos[0] < 0
            or env.height <= to_pos[0]
            or to_pos[1] < 0
            or env.width <= to_pos[1]
            or env.lakes[to_pos]
        ):
            continue

        from_pos2 = (3 + 4 - from_pos[0], from_pos[1])
        to_pos2 = (from_pos2[0] - direction[0], from_pos2[1] - direction[1])
        if (
            from_pos2[0] < 0
            or env.height <= from_pos2[0]
            or from_pos2[1] < 0
            or env.width <= from_pos2[1]
            or env.lakes[from_pos2]
        ):
            continue
        if (
            to_pos2[0] < 0
            or env.height <= to_pos2[0]
            or to_pos2[1] < 0
            or env.width <= to_pos2[1]
            or env.lakes[to_pos2]
        ):
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

        assert env.two_square_detector.validate_move(
            Player.RED, piece, to_pos, from_pos
        )
        assert env.two_square_detector.validate_move(
            Player.BLUE, piece, to_pos, from_pos
        )
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
    ),
)
def test_no_violation_interrupt_sync(
    env_5x5: Callable[[Any], StrategoEnv], pices, from_pos
):
    env = env_5x5(pices)
    move_direction1 = np.array([-1, 1, 0, 0])
    move_direction2 = np.roll(move_direction1, 2)
    for direction in zip(move_direction1, move_direction2):
        to_pos = (from_pos[0] + direction[0], from_pos[1] + direction[1])
        if (
            to_pos[0] < 0
            or env.height <= to_pos[0]
            or to_pos[1] < 0
            or env.width <= to_pos[1]
            or env.lakes[to_pos]
        ):
            continue

        from_pos2 = (from_pos[0], from_pos[1] + 1)
        to_pos2 = (from_pos2[0] + direction[0], from_pos2[1] + direction[1])
        if (
            from_pos2[0] < 0
            or env.height <= from_pos2[0]
            or from_pos2[1] < 0
            or env.width <= from_pos2[1]
            or from_pos2 == to_pos
            or env.lakes[from_pos2]
        ):
            continue

        if (
            to_pos2[0] < 0
            or env.height <= to_pos2[0]
            or to_pos2[1] < 0
            or env.width <= to_pos2[1]
            or from_pos == to_pos2
            or env.lakes[to_pos2]
        ):
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
        [
            SCOUT_ONLY,
        ],
        itertools.product(range(3, 5), range(5)),
    ),
)
def test_scout_move_entire_row(env_5x5: Callable[[Any], StrategoEnv], pices, from_pos):
    env = env_5x5(pices)
    move_direction1 = np.array([-1, 1, 0, 0])
    move_direction2 = np.roll(move_direction1, 2)
    for direction in zip(move_direction1, move_direction2):
        to_pos = from_pos
        while True:
            _to_pos = (to_pos[0] + direction[0], to_pos[1] + direction[1])
            if (
                _to_pos[0] < 0
                or env.height <= _to_pos[0]
                or _to_pos[1] < 0
                or env.width <= _to_pos[1]
                or env.lakes[_to_pos]
            ):
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

        assert not env.two_square_detector.validate_move(
            Player.RED, piece, to_pos, from_pos
        )
        assert not env.two_square_detector.validate_move(
            Player.BLUE, piece, to_pos, from_pos
        )
        assert len(env.two_square_detector.p1) == 3
        assert len(env.two_square_detector.p2) == 3

        repeat_twice(move_bwd, env, from_pos, to_pos)
        validate_move(env, Piece.EMPTY, to_pos, from_pos, maybe_occupied=True)
        assert len(env.two_square_detector.p1) == 1
        assert len(env.two_square_detector.p2) == 1


class TestScoutPositionalMovement:

    def test_range_narrowing_end_pos(self, env_5x5: Callable[[Any], StrategoEnv]):
        env = env_5x5(SCOUT_ONLY)
        pos1 = (4, 1)  # starting position
        pos2 = (
            4,
            3,
        )  # final_position: repeat movement in the same direction (pos3 -> pos4 -> pos2)
        pos3 = (4, 1)
        pos4 = (4, 2)
        pos5 = (4, 0)  # final_position: one cell to the left of the starting position
        for final_position in (pos2, pos5):
            env.reset()
            repeat_twice(env.step, pos1)
            piece = Piece(env.board[pos1])
            assert piece.value > Piece.LAKE.value
            assert env.game_phase == GamePhase.SELECT
            repeat_twice(move_fwd, env, pos1, pos2)
            validate_move(env, piece, pos1, pos2)

            repeat_twice(move_bwd, env, pos3, pos2)
            validate_move(env, piece, pos2, pos3)

            repeat_twice(move_fwd, env, pos3, pos4)
            validate_move(env, piece, pos3, pos4)

            assert not env.two_square_detector.validate_move(
                Player.RED, piece, pos4, pos3
            )
            assert not env.two_square_detector.validate_move(
                Player.BLUE, piece, pos4, pos3
            )
            assert len(env.two_square_detector.p1) == 3
            assert len(env.two_square_detector.p2) == 3

            assert env.two_square_detector.validate_move(
                Player.RED, piece, pos4, final_position
            )
            assert env.two_square_detector.validate_move(
                Player.BLUE, piece, pos4, final_position
            )
            repeat_twice(move_fwd, env, pos4, final_position)
            validate_move(env, piece, pos4, final_position)
            assert len(env.two_square_detector.p1) == 1
            assert len(env.two_square_detector.p2) == 1

    def test_range_narrowing_start_pos(self, env_5x5: Callable[[Any], StrategoEnv]):
        env = env_5x5(SCOUT_ONLY)
        pos1 = (4, 1)  # starting position;
        pos2 = (4, 3)
        pos3 = (4, 2)
        pos4 = (4, 3)
        pos5 = (4, 0)  # final_position: one cell to the left of the starting position
        pos6 = (
            4,
            4,
        )  # final_position: one cell to the right of the intermediate position (pos2, pos4)
        for final_position in (pos5, pos6):
            env.reset()
            repeat_twice(env.step, pos1)
            piece = Piece(env.board[pos1])
            assert piece.value > Piece.LAKE.value
            assert env.game_phase == GamePhase.SELECT
            repeat_twice(move_fwd, env, pos1, pos2)
            validate_move(env, piece, pos1, pos2)

            repeat_twice(move_bwd, env, pos3, pos2)
            validate_move(env, piece, pos2, pos3)

            repeat_twice(move_fwd, env, pos3, pos4)
            validate_move(env, piece, pos3, pos4)

            assert not env.two_square_detector.validate_move(
                Player.RED, piece, pos4, pos3
            )
            assert not env.two_square_detector.validate_move(
                Player.BLUE, piece, pos4, pos3
            )
            assert not env.two_square_detector.validate_move(
                Player.RED, piece, pos4, pos1
            )
            assert not env.two_square_detector.validate_move(
                Player.BLUE, piece, pos4, pos1
            )
            assert len(env.two_square_detector.p1) == 3
            assert len(env.two_square_detector.p2) == 3

            assert env.two_square_detector.validate_move(
                Player.RED, piece, pos4, final_position
            )
            assert env.two_square_detector.validate_move(
                Player.BLUE, piece, pos4, final_position
            )
            repeat_twice(move_fwd, env, pos4, final_position)
            validate_move(env, piece, pos4, final_position)
            assert len(env.two_square_detector.p1) == 1
            assert len(env.two_square_detector.p2) == 1

    def test_range_narrowing_start_end_pos(self, env_5x5: Callable[[Any], StrategoEnv]):
        env = env_5x5(SCOUT_ONLY)
        pos1 = (4, 1)  # starting position;
        pos2 = (
            4,
            4,
        )  # final_position: repeat movement in the same direction (pos3 -> pos4 -> pos2)
        pos3 = (4, 2)
        pos4 = (4, 3)
        pos5 = (4, 0)  # final_position: one cell to the left of the starting position
        for final_position in (pos2, pos5):
            env.reset()
            repeat_twice(env.step, pos1)
            piece = Piece(env.board[pos1])
            assert piece.value > Piece.LAKE.value
            assert env.game_phase == GamePhase.SELECT
            repeat_twice(move_fwd, env, pos1, pos2)
            validate_move(env, piece, pos1, pos2)

            repeat_twice(move_bwd, env, pos3, pos2)
            validate_move(env, piece, pos2, pos3)

            repeat_twice(move_fwd, env, pos3, pos4)
            validate_move(env, piece, pos3, pos4)

            assert not env.two_square_detector.validate_move(
                Player.RED, piece, pos4, pos3
            )
            assert not env.two_square_detector.validate_move(
                Player.BLUE, piece, pos4, pos3
            )
            assert not env.two_square_detector.validate_move(
                Player.RED, piece, pos4, pos1
            )
            assert not env.two_square_detector.validate_move(
                Player.BLUE, piece, pos4, pos1
            )
            assert len(env.two_square_detector.p1) == 3
            assert len(env.two_square_detector.p2) == 3

            assert env.two_square_detector.validate_move(
                Player.RED, piece, pos4, final_position
            )
            assert env.two_square_detector.validate_move(
                Player.BLUE, piece, pos4, final_position
            )
            repeat_twice(move_fwd, env, pos4, final_position)
            validate_move(env, piece, pos4, final_position)
            assert len(env.two_square_detector.p1) == 1
            assert len(env.two_square_detector.p2) == 1

    def test_range_extension_end_pos(self, env_5x5: Callable[[Any], StrategoEnv]):
        env = env_5x5(SCOUT_ONLY)
        pos1 = (4, 1)  # starting position
        pos2 = (4, 2)
        pos3 = (4, 1)
        pos4 = (4, 3)
        repeat_twice(env.step, pos1)
        piece = Piece(env.board[pos1])
        assert piece.value > Piece.LAKE.value
        assert env.game_phase == GamePhase.SELECT
        repeat_twice(move_fwd, env, pos1, pos2)
        validate_move(env, piece, pos1, pos2)

        repeat_twice(move_bwd, env, pos3, pos2)
        validate_move(env, piece, pos2, pos3)

        repeat_twice(move_fwd, env, pos3, pos4)
        validate_move(env, piece, pos3, pos4)

        assert env.two_square_detector.validate_move(Player.RED, piece, pos4, pos3)
        assert env.two_square_detector.validate_move(Player.BLUE, piece, pos4, pos3)
        repeat_twice(move_fwd, env, pos4, pos3)
        validate_move(env, piece, pos4, pos3)
        assert len(env.two_square_detector.p1) == 2  # (4, 1) -> (4, 3) -> (4, 1)
        assert len(env.two_square_detector.p2) == 2

    def test_range_extension_start_pos(self, env_5x5: Callable[[Any], StrategoEnv]):
        env = env_5x5(SCOUT_ONLY)
        pos1 = (4, 1)  # starting position
        pos2 = (4, 3)
        pos3 = (4, 0)
        pos4 = (4, 3)
        repeat_twice(env.step, pos1)
        piece = Piece(env.board[pos1])
        assert piece.value > Piece.LAKE.value
        assert env.game_phase == GamePhase.SELECT
        repeat_twice(move_fwd, env, pos1, pos2)
        validate_move(env, piece, pos1, pos2)

        repeat_twice(move_bwd, env, pos3, pos2)
        validate_move(env, piece, pos2, pos3)

        repeat_twice(move_fwd, env, pos3, pos4)
        validate_move(env, piece, pos3, pos4)

        assert env.two_square_detector.validate_move(Player.RED, piece, pos4, pos3)
        assert env.two_square_detector.validate_move(Player.BLUE, piece, pos4, pos3)
        repeat_twice(move_fwd, env, pos4, pos3)
        validate_move(env, piece, pos4, pos3)
        assert (
            len(env.two_square_detector.p1) == 3
        )  # (4, 3) -> (4, 0) -> (4, 3) -> (4, 0)
        assert len(env.two_square_detector.p2) == 3

    def test_range_extension_start_end_pos(self, env_5x5: Callable[[Any], StrategoEnv]):
        env = env_5x5(SCOUT_ONLY)
        pos1 = (4, 1)  # starting position
        pos2 = (4, 2)
        pos3 = (4, 0)
        pos4 = (4, 3)
        repeat_twice(env.step, pos1)
        piece = Piece(env.board[pos1])
        assert piece.value > Piece.LAKE.value
        assert env.game_phase == GamePhase.SELECT
        repeat_twice(move_fwd, env, pos1, pos2)
        validate_move(env, piece, pos1, pos2)

        repeat_twice(move_bwd, env, pos3, pos2)
        validate_move(env, piece, pos2, pos3)

        repeat_twice(move_fwd, env, pos3, pos4)
        validate_move(env, piece, pos3, pos4)

        assert env.two_square_detector.validate_move(Player.RED, piece, pos4, pos3)
        assert env.two_square_detector.validate_move(Player.BLUE, piece, pos4, pos3)
        repeat_twice(move_fwd, env, pos4, pos3)
        validate_move(env, piece, pos4, pos3)
        assert len(env.two_square_detector.p1) == 2  # (4, 0) -> (4, 3) -> (4, 0)
        assert len(env.two_square_detector.p2) == 2

    def test_movement_sample(self, env_10x10: Callable[[Any], StrategoEnv]):
        env = env_10x10(SCOUT_ONLY)
        from_pos = (6, 1)
        repeat_twice(env.step, from_pos)
        piece = Piece(env.board[from_pos])
        assert piece.value > Piece.LAKE.value
        assert env.game_phase == GamePhase.SELECT

        event_list = [
            (False, (6, 6)),
            (False, (6, 1)),
            (False, (6, 6)),
            (True, (6, 1)),  # equal jumps
            (False, (6, 0)),
            (False, (6, 6)),
            (False, (6, 4)),
            (True, (6, 6)),  # narrowing start pos
            (False, (6, 7)),
            (False, (6, 4)),
            (False, (6, 6)),
            (True, (6, 4)),  # narrowing end pos
            (False, (6, 3)),
            (False, (6, 6)),
            (False, (6, 4)),
            (True, (6, 6)),  # narrowing start pos
            (False, (6, 7)),
            (False, (6, 2)),
            (False, (6, 4)),
            (False, (6, 0)),
            (False, (6, 6)),
            (False, (6, 4)),
            (False, (6, 7)),
            (False, (6, 4)),
            (False, (6, 6)),
            (True, (6, 4)),  # narrowing end pos
            (False, (6, 3)),
        ]

        for need_check, to_pos in event_list:
            if not need_check:
                repeat_twice(move_fwd, env, from_pos, to_pos)
                validate_move(env, piece, from_pos, to_pos)
                from_pos = to_pos
            else:
                assert not env.two_square_detector.validate_move(
                    Player.RED, piece, from_pos, to_pos
                )
                assert not env.two_square_detector.validate_move(
                    Player.BLUE, piece, from_pos, to_pos
                )
