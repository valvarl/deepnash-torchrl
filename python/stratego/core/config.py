from __future__ import annotations

from abc import ABC
from collections.abc import Iterable
from enum import Enum

import numpy as np

from stratego.core.primitives import Piece, Pos

PIECES_NUM_ORIGINAL = {
    Piece.FLAG: 1,
    Piece.BOMB: 6,
    Piece.SPY: 1,
    Piece.SCOUT: 8,
    Piece.MINER: 5,
    Piece.SERGEANT: 4,
    Piece.LIEUTENANT: 4,
    Piece.CAPTAIN: 4,
    Piece.MAJOR: 3,
    Piece.COLONEL: 2,
    Piece.GENERAL: 1,
    Piece.MARSHAL: 1,
}

PIECES_NUM_BARRAGE = {
    Piece.FLAG: 1,
    Piece.BOMB: 1,
    Piece.SPY: 1,
    Piece.SCOUT: 2,
    Piece.MINER: 1,
    Piece.SERGEANT: 0,
    Piece.LIEUTENANT: 0,
    Piece.CAPTAIN: 0,
    Piece.MAJOR: 0,
    Piece.COLONEL: 0,
    Piece.GENERAL: 1,
    Piece.MARSHAL: 1,
}

PLACES_TO_DEPLOY_RED_ORIGINAL = [
    ((6, 0), (9, 9)),
]
PLACES_TO_DEPLOY_BLUE_ORIGINAL = [
    ((0, 0), (3, 9)),
]
LAKES_ORIGINAL = [((4, 2), (5, 3)), ((4, 6), (5, 7))]


class GameMode(Enum):
    CUSTOM = 0
    ORIGINAL = 1
    BARRAGE = 2


class StrategoConfigBase(ABC):
    def __init__(
        self,
        height: int,
        width: int,
        p1_pieces: dict[Piece, int],
        p2_pieces: dict[Piece, int] | None = None,
        lakes: Iterable[tuple[Pos, Pos]] | None = None,
        p1_places_to_deploy: Iterable[tuple[Pos, Pos]] | None = None,
        p2_places_to_deploy: Iterable[tuple[Pos, Pos]] | None = None,
        lakes_mask: np.ndarray | None = None,
        p1_deploy_mask: np.ndarray | None = None,
        p2_deploy_mask: np.ndarray | None = None,
        total_moves_limit: int = 2000,
        moves_since_attack_limit: int | None = 200,
        observed_history_entries: int = 40,
        allow_competitive_deploy: bool = False,
        game_mode: GameMode = GameMode.CUSTOM,
    ):
        pass

    @classmethod
    def from_game_mode(cls, game_mode: GameMode) -> StrategoConfigBase:
        if game_mode == GameMode.ORIGINAL:
            return cls(
                height=10,
                width=10,
                p1_pieces=PIECES_NUM_ORIGINAL,
                p1_places_to_deploy=PLACES_TO_DEPLOY_RED_ORIGINAL,
                p2_places_to_deploy=PLACES_TO_DEPLOY_BLUE_ORIGINAL,
                lakes=LAKES_ORIGINAL,
            )
        elif game_mode == GameMode.BARRAGE:
            return cls(
                height=10,
                width=10,
                p1_pieces=PIECES_NUM_ORIGINAL,
                p1_places_to_deploy=PLACES_TO_DEPLOY_RED_ORIGINAL,
                p2_places_to_deploy=PLACES_TO_DEPLOY_BLUE_ORIGINAL,
                lakes=LAKES_ORIGINAL,
            )
        else:
            raise ValueError(f"Unknown game mode: {game_mode}")


class StrategoConfig(StrategoConfigBase):
    def __init__(
        self,
        height: int,
        width: int,
        p1_pieces: dict[Piece, int],
        p2_pieces: dict[Piece, int] | None = None,
        lakes: Iterable[tuple[Pos, Pos]] | None = None,
        p1_places_to_deploy: Iterable[tuple[Pos, Pos]] | None = None,
        p2_places_to_deploy: Iterable[tuple[Pos, Pos]] | None = None,
        lakes_mask: np.ndarray | None = None,
        p1_deploy_mask: np.ndarray | None = None,
        p2_deploy_mask: np.ndarray | None = None,
        total_moves_limit: int = 2000,
        moves_since_attack_limit: int | None = 200,
        observed_history_entries: int = 40,
        allow_competitive_deploy: bool = False,
        game_mode: GameMode = GameMode.CUSTOM,
    ):
        self.height = height
        self.width = width

        self.p1_deploy_mask = self._resolve_mask(p1_places_to_deploy, p1_deploy_mask)
        self.p2_deploy_mask = self._resolve_mask(p2_places_to_deploy, p2_deploy_mask)

        self.p1_pieces = p1_pieces
        self.p1_pieces_num = self._pieces_to_array(p1_pieces)
        if p2_pieces is not None:
            self.p2_pieces = p2_pieces
            self.p2_pieces_num = self._pieces_to_array(p2_pieces)
        else:
            self.p2_pieces = p1_pieces.copy()
            self.p2_pieces_num = self.p1_pieces_num.copy()

        self.allowed_pieces = np.arange(len(self.p1_pieces_num))[
            (self.p1_pieces_num != 0) | (self.p2_pieces_num != 0)
        ]

        self.lakes_mask = self._resolve_mask(lakes, lakes_mask)

        self.total_moves_limit = total_moves_limit
        self.moves_since_attack_limit = moves_since_attack_limit
        if moves_since_attack_limit is None:
            self.moves_since_attack_limit = self.total_moves_limit
        self.observed_history_entries = observed_history_entries

        self.allow_competitive_deploy = allow_competitive_deploy
        valid, msg = self._validate()
        if not valid:
            raise ValueError(msg)

        self.game_mode = game_mode

    def rot90(self, k: int = 1) -> StrategoConfig:
        return StrategoConfig(
            height=self.height if k % 2 == 0 else self.width,
            width=self.width if k % 2 == 0 else self.heigh,
            p1_pieces=self.p1_pieces,
            p2_pieces=self.p2_pieces,
            p1_deploy_mask=np.rot90(self.p1_deploy_mask, k),
            p2_deploy_mask=np.rot90(self.p2_deploy_mask, k),
            lakes_mask=np.rot90(self.lakes_mask, k),
            total_moves_limit=self.total_moves_limit,
            moves_since_attack_limit=self.moves_since_attack_limit,
            observed_history_entries=self.observed_history_entries,
            allow_competitive_deploy=self.allow_competitive_deploy,
            game_mode=self.game_mode,
        )

    def _resolve_mask(
        self, positions: Iterable[tuple[Pos, Pos]] | None, mask: np.ndarray | None
    ) -> np.ndarray:
        mask_resolved, valid, msg = None, False, ""
        if positions is None and mask is None:
            msg = "Both 'positions' and 'mask' are None. At least one must be provided."
        elif positions is not None and mask is not None:
            msg = "Both 'positions' and 'mask' are provided. Only one should be specified."
        elif positions is not None:
            mask_resolved = self._make_mask(positions)
            valid = True
        else:
            mask_resolved = mask
            valid = True
        if not valid:
            raise AttributeError(msg)
        return mask_resolved

    def _make_mask(self, positions: Iterable[tuple[Pos, Pos]] | None) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=bool)
        if positions is None:
            return mask
        for (y1, x1), (y2, x2) in positions:
            y1, y2 = (y1, y2) if y1 < y2 else (y2, y1)
            x1, x2 = (x1, x2) if x1 < x2 else (x2, x1)
            mask[y1 : y2 + 1, x1 : x2 + 1] = 1
        return mask

    def _pieces_to_array(self, pieces_num: dict[Piece, int]) -> np.ndarray:
        pieces = np.zeros((len(Piece),), dtype=np.int64)
        for piece, num in pieces_num.items():
            pieces[piece.value] = num
        return pieces

    def _validate(self) -> tuple[bool, str]:
        if (self.p1_deploy_mask & self.lakes_mask).any():
            return False, "Player 1's deployment overlaps with lakes."
        elif (self.p2_deploy_mask & self.lakes_mask).any():
            return False, "Player 2's deployment overlaps with lakes."

        if not self.allow_competitive_deploy:
            if (self.p1_deploy_mask & self.p2_deploy_mask).any():
                return False, "Player 1's and Player 2's deployments overlap."
            if self.p1_deploy_mask.sum() < self.p1_pieces_num.sum():
                return False, "Player 1 has fewer deployment spots than pieces."
            if self.p2_deploy_mask.sum() < self.p2_pieces_num.sum():
                return False, "Player 2 has fewer deployment spots than pieces."

        else:
            xor = np.logical_xor(self.p1_deploy_mask, self.p2_deploy_mask)
            p1_own_places = (xor & self.p1_deploy_mask).sum()
            p2_own_places = (xor & self.p2_deploy_mask).sum()
            if (
                self.p1_pieces_num.sum()
                - p1_own_places
                + self.p2_pieces_num.sum()
                - p2_own_places
                > (self.p1_deploy_mask & self.p2_deploy_mask).sum()
            ):
                return (
                    False,
                    "Total number of pieces exceeds available shared deployment spots.",
                )

        if self.total_moves_limit <= 0:
            return False, "Total moves limit must be greater than 0."

        if self.moves_since_attack_limit <= 0:
            return False, "Moves since last attack limit must be greater than 0."

        if self.observed_history_entries < 0:
            return False, "Observed history entries cannot be negative."

        return True, "Validation successful."
