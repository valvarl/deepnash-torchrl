from typing import Iterable
import numpy as np
from stratego.core.config import GameMode, StrategoConfigBase
from stratego.core.primitives import Piece, Pos

from stratego.cpp import stratego_cpp as sp


class StrategoConfigCpp(StrategoConfigBase):
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
        p1_pieces_cpp = self._pieces_to_cpp(p1_pieces)
        p2_pieces_cpp = self._pieces_to_cpp(p2_pieces)

        lakes_mask_cpp = self._make_mask_cpp(lakes_mask, height, width)
        p1_deploy_mask_cpp = self._make_mask_cpp(p1_deploy_mask, height, width)
        p2_deploy_mask_cpp = self._make_mask_cpp(p2_deploy_mask, height, width)

        game_mode_cpp = self._game_mode_to_cpp(game_mode)

        args = {
            "height": height,
            "width": width,
            "p1_pieces": p1_pieces_cpp,
            "total_moves_limit": total_moves_limit,
            "moves_since_attack_limit": moves_since_attack_limit,
            "observed_history_entries": observed_history_entries,
            "allow_competitive_deploy": allow_competitive_deploy,
            "game_mode": game_mode_cpp,
        }

        if p2_pieces_cpp is not None:
            args["p2_pieces"] = p2_pieces_cpp
        if lakes is not None:
            args["lakes"] = lakes
        if p1_places_to_deploy is not None:
            args["p1_places_to_deploy"] = p1_places_to_deploy
        if p2_places_to_deploy is not None:
            args["p2_places_to_deploy"] = p2_places_to_deploy
        if lakes_mask is not None:
            args["lakes_mask"] = lakes_mask_cpp
        if p1_deploy_mask is not None:
            args["p1_deploy_mask"] = p1_deploy_mask_cpp
        if p2_deploy_mask is not None:
            args["p2_deploy_mask"] = p2_deploy_mask_cpp

        self._config_cpp = sp.StrategoConfig(**args)

    def _pieces_to_cpp(self, pieces: dict[Piece, int]) -> dict[sp.Piece, int] | None:
        if pieces is None:
            return None
        return {sp.Piece(piece.value): num for piece, num in pieces.items()}

    def _game_mode_to_cpp(self, game_mode: GameMode) -> sp.GameMode:
        return sp.GameMode(game_mode.value)

    def _make_mask_cpp(self, mask: np.ndarray | None, height: int, width: int):
        if mask is None:
            return None

        mask_cpp = [False for _ in range(height * width)]
        for y in range(height):
            for x in range(width):
                mask_cpp[y * width + x] = bool(mask[y][x])
        return mask_cpp

    @property
    def height(self):
        return self._config_cpp.height

    @property
    def width(self):
        return self._config_cpp.width

    @property
    def observed_history_entries(self):
        return self._config_cpp.observed_history_entries

    @property
    def allowed_pieces(self):
        return self._config_cpp.allowed_pieces
