from abc import ABC, abstractmethod

import numpy as np
from gymnasium import Env, spaces
import pygame

from stratego.core.config import StrategoConfigBase
from stratego.core.masked_multi_descrete import MaskedMultiDiscrete
from stratego.core.primitives import Piece, Player


# PyGame Rendering Constants
WINDOW_SIZE = 800


class StrategoEnvBase(Env):
    metadata = {"render_modes": [None, "human"]}

    def __init__(
        self, config: StrategoConfigBase | None = None, render_mode: str | None = None
    ):
        self.render_mode = render_mode
        if render_mode not in [None, "human", "rgb_array"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.window = None

        self.observation_space: spaces.Dict
        self.action_space: MaskedMultiDiscrete

    @abstractmethod
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.window = None

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def get_info(self):
        pass

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))

        cell_size = (
            WINDOW_SIZE // self.board.shape[0],
            WINDOW_SIZE // self.board.shape[1],
        )

        # Calculate the number of cells in each direction
        num_cells_r = self.board.shape[0]
        num_cells_c = self.board.shape[1]

        board = np.copy(self.board)
        if self.player == Player.BLUE:
            board = np.rot90(board, 2) * -1

        font = pygame.font.Font(None, 80)
        # Draw the grid
        for r in range(num_cells_r):
            for c in range(num_cells_c):
                rect = pygame.Rect(
                    c * cell_size[0], r * cell_size[1], cell_size[0], cell_size[1]
                )
                rect_center = rect.center
                if abs(board[r][c]) == Piece.LAKE.value:
                    pygame.draw.rect(self.window, (158, 194, 230), rect)
                    text = None
                elif board[r][c] >= Piece.FLAG.value:
                    pygame.draw.rect(self.window, (217, 55, 58), rect)
                    render_text = (
                        "F"
                        if board[r][c] == Piece.FLAG.value
                        else (
                            "B"
                            if board[r][c] == Piece.BOMB.value
                            else str(board[r][c] - 3)
                        )
                    )
                    text = font.render(render_text, True, (255, 255, 255))
                elif board[r][c] <= -Piece.FLAG.value:
                    pygame.draw.rect(self.window, (24, 118, 181), rect)
                    render_text = (
                        "*"
                        if board[r][c] == -Piece.FLAG.value
                        else "*" if board[r][c] == -Piece.BOMB.value else "*"
                    )  # str(-(board[r][c] + 3))
                    text = font.render(render_text, True, (255, 255, 255))
                else:
                    pygame.draw.rect(self.window, (242, 218, 180), rect)
                    text = None

                pygame.draw.rect(self.window, (255, 255, 255), rect, width=3)
                if text is not None:
                    text_rect = text.get_rect(center=rect_center)
                    self.window.blit(text, text_rect)

        pygame.event.pump()
        pygame.display.update()
