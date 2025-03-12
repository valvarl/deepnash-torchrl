

from collections.abc import Iterable
from enum import Enum

import numpy as np

from stratego_gym.envs.primitives import Piece, Pos

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

PLACES_TO_DEPLOY_RED_ORIGINAL = [((6, 0), (9, 9)),]
PLACES_TO_DEPLOY_BLUE_ORIGINAL = [((0, 0), (3, 9)),]
LAKES_ORIGINAL = [((4, 2), (5, 3)), ((4, 6), (5, 7))]

class GameMode(Enum):
    ORIGINAL = 0
    BARRAGE = 1

class StrategoConfig:
    def __init__(
        self,
        height: int,
        width: int,
        p1_pieces_num: dict[Piece, int],
        p2_pieces_num: dict[Piece, int] | None = None,
        lakes: Iterable[tuple[Pos, Pos]] | None = None,
        p1_places_to_deploy: Iterable[tuple[Pos, Pos]] | None = None,
        p2_places_to_deploy: Iterable[tuple[Pos, Pos]] | None = None,
        lakes_mask: np.ndarray | None = None,
        p1_deploy_mask: np.ndarray | None = None,
        p2_deploy_mask: np.ndarray | None = None,
        total_moves_limit: int = 2000,
        moves_since_attack_limit: int | None = 200,
        observed_history_entries: int = 0,
        allow_competitive_deploy: bool = False,
        game_mode: GameMode | None = None,
    ):
        self.height = height
        self.width = width

        self.p1_deploy_mask = self._resolve_mask(p1_places_to_deploy, p1_deploy_mask)        
        self.p2_deploy_mask = self._resolve_mask(p2_places_to_deploy, p2_deploy_mask)
    
        self.p1_pieces_num = self._pieces_to_array(p1_pieces_num)
        if p2_pieces_num is not None:
            self.p2_pieces_num = self._pieces_to_array(p2_pieces_num)
        else:
            self.p2_pieces_num = self.p1_pieces_num.copy()

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
        
    def _resolve_mask(self, positions: Iterable[tuple[Pos, Pos]] | None, mask: np.ndarray | None) -> np.ndarray:
        mask_resolved, valid, msg = None, False, ""
        if positions is None and mask is None:
            msg = ""
        elif positions is not None and mask is not None:
            msg = ""
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
            mask[y1: y2 + 1, x1: x2 + 1] = 1
        return mask
    
    def _pieces_to_array(self, pieces_num: dict[Piece, int]) -> np.ndarray:
        pieces = np.zeros((Piece.unique_pieces_num(),), dtype=np.int64)
        for piece, num in pieces_num.items():
            if piece.value < Piece.FLAG.value:
                raise ValueError("")
            pieces[piece.value - Piece.FLAG.value] = num
        return pieces
    
    def _validate(self) -> tuple[bool, str]:
        if (self.p1_deploy_mask & self.lakes_mask).any():
            return False, ""
        elif (self.p2_deploy_mask & self.lakes_mask).any():
            return False, ""
        
        if not self.allow_competitive_deploy:
            if (self.p1_deploy_mask & self.p2_deploy_mask).any():
                return False, ""
            if self.p1_deploy_mask.sum() < self.p1_pieces_num.sum():
                return False, ""
            if self.p2_deploy_mask.sum() < self.p2_pieces_num.sum():
                return False, ""
        
        else:
            xor = np.logical_xor(self.p1_deploy_mask, self.p2_deploy_mask)
            p1_own_places = (xor & self.p1_deploy_mask).sum()
            p2_own_places = (xor & self.p2_deploy_mask).sum()
            if self.p1_pieces_num.sum() - p1_own_places + self.p2_pieces_num.sum() - p2_own_places > \
            (self.p1_deploy_mask & self.p2_deploy_mask).sum():
                return False, ""        
            
        if self.total_moves_limit <= 0:
            return False, ""
        
        if self.moves_since_attack_limit <= 0:
            return False, ""
            
        if self.observed_history_entries < 0:
            return False, ""
        
        return True, ""
    
    @classmethod
    def from_game_mode(cls, game_mode: GameMode):
        if game_mode == GameMode.ORIGINAL:
            return cls(
                height=10,
                width=10,
                p1_pieces_num=PIECES_NUM_ORIGINAL,
                p1_places_to_deploy=PLACES_TO_DEPLOY_RED_ORIGINAL,
                p2_places_to_deploy=PLACES_TO_DEPLOY_BLUE_ORIGINAL,
                lakes=LAKES_ORIGINAL,
            )
        elif game_mode == GameMode.BARRAGE:
            return cls(
                height=10,
                width=10,
                p1_pieces_num=PIECES_NUM_ORIGINAL,
                p1_places_to_deploy=PLACES_TO_DEPLOY_RED_ORIGINAL,
                p2_places_to_deploy=PLACES_TO_DEPLOY_BLUE_ORIGINAL,
                lakes=LAKES_ORIGINAL,
            )
        else:
            raise ValueError(f"Unknown game mode: {game_mode}")
