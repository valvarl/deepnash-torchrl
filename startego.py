from enum import Enum, auto
import random

import numpy as np
from gymnasium import Env, spaces
import pygame

from config import StrategoConfig, GameMode
from masked_multi_descrete import MaskedMultiDiscrete
from primitives import Piece, Pos


class GamePhase(Enum):
    TERMINAL = auto()
    DEPLOYMENT = auto()
    SELECT = auto()
    MOVE = auto()


class Player(Enum):
    RED = 1
    BLUE = -1


class StrategoEnv(Env):
    metadata = {"render_modes": [None, "human"]}

    def __init__(self, config: StrategoConfig | None = None, render_mode: str | None = None):
        
        self.config = config
        if config is None:
            self.config = StrategoConfig.from_game_mode(GameMode.ORIGINAL)

        self.render_mode = render_mode
        if render_mode not in [None, "human", "rgb_array"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        
        self.window = None
        self.clock = None

        self.height = self.config.height
        self.width = self.config.width

        self.game_phase = GamePhase.TERMINAL
        self.player = Player.RED

        self.board = None
        self.p1_pieces = None
        self.p2_pieces = None
        self.p1_deploy_mask = None
        self.p2_deploy_mask = None

        self.p1_last_selected = None
        self.p2_last_selected = None

        self.max_total_moves: int | None = None
        self.max_moves_since_attack: int | None = None
        self.observed_history_entries: int | None = None

        self.draw_conditions = {"total_moves": 0, "moves_since_attack": 0}     

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        
    def _get_observation_space(self):
        observation_channels = ((self.config.p1_pieces_num != 0) | (self.config.p2_pieces_num != 0)).sum() * 3 + \
        self.config.observed_history_entries + 6

        shape = (observation_channels, self.config.height, self.config.width)
        mask_shape = (self.config.height, self.config.width)

        return spaces.Dict({
            "obs": spaces.Box(low=-3, high=1, shape=shape, dtype=np.float64),
            "action_mask": spaces.Box(low=0, high=1, shape=mask_shape, dtype=np.int64)
        })
    
    def _get_action_space(self):
        if getattr(self, "action_space", None) is None:
            return spaces.MultiDiscrete((self.height, self.width), dtype=np.int64)
        else:
            return self.action_space
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.window = None
        self.clock = None

        if getattr(self, "height", None) != self.config.height or getattr(self, "width", None) != self.config.width:
            del self.action_space
            self.height = self.config.height
            self.width = self.config.width
            self.action_space = self._get_action_space()

        self.game_phase = GamePhase.DEPLOYMENT
        self.player = Player.RED
        self.p1_deploy_idx = 0
        self.p2_deploy_idx = 0

        self.max_total_moves = self.config.max_total_moves
        self.max_moves_since_attack = self.config.max_moves_since_attack
        self.observed_history_entries = self.config.observed_history_entries
        self.draw_conditions = {"total_moves": 0, "moves_since_attack": 0}
        
        self.generate_board()
        return self.generate_env_state(), self.get_info()
    
    def generate_board(self):
        self.board = np.zeros(self.height, self.width)
        self.board[self.config.lakes_mask] = Piece.LAKE

        self.p1_deploy_mask = np.copy(self.config.p1_deploy_mask)
        self.p2_deploy_mask = np.copy(self.config.p2_deploy_mask)

        self.p1_pieces = np.arange(len(self.config.p1_pieces_num))[self.config.p1_pieces_num != 0]
        self.p2_pieces = np.arange(len(self.config.p2_pieces_num))[self.config.p2_pieces_num != 0]

        self.p1_movable_pieces = self.p1_pieces[~np.isin(self.p1_pieces, [Piece.FLAG, Piece.BOMB])]
        self.p2_movable_pieces = self.p2_pieces[~np.isin(self.p2_pieces, [Piece.FLAG, Piece.BOMB])]

        self.p1_public_obs_info = np.zeros((3, self.height, self.width))
        self.p2_public_obs_info = np.zeros((3, self.height, self.width))
        self.p1_public_obs_info[0, self.p2_deploy_mask] = 1
        self.p2_public_obs_info[0, self.p1_deploy_mask] = 1

        self.p1_unrevealed = np.copy(self.config.p1_pieces_num)
        self.p2_unrevealed = np.copy(self.config.p1_pieces_num)

        self.p1_observed_moves = np.zeros((self.config.observed_history_entries, self.height, self.width))
        self.p2_observed_moves = np.zeros((self.config.observed_history_entries, self.height, self.width))

    def generate_observation(self):
        lakes = (np.abs(self.board) == Piece.LAKE)[None, :]
        observation_channels = ((self.config.p1_pieces_num != 0) | (self.config.p2_pieces_num != 0)).sum()
        private_obs = np.eye(observation_channels)[np.where(self.board > Piece.LAKE.value, self.board - 2, 0)]

        if self.game_phase == GamePhase.DEPLOYMENT:
            public_obs = np.zeros((observation_channels, self.height, self.width))
            opp_public_obs = np.zeros((observation_channels, self.height, self.width))
            moves_obs = np.zeros_like(self.p1_observed_moves)
        else:
            public_obs1 = self.get_public_obs(self.p1_public_obs_info, self.p1_unrevealed, self.p1_pieces, self.p1_movable_pieces)
            public_obs2 = self.get_public_obs(self.p2_public_obs_info, self.p2_unrevealed, self.p2_pieces, self.p2_movable_pieces)
            public_obs = public_obs1 if self.player == 1 else public_obs2
            opp_public_obs = public_obs2 if self.player == 1 else public_obs1
            moves_obs = self.p1_observed_moves if self.player == 1 else self.p2_observed_moves

        scalar_obs = np.ones((4, self.height, self.width))
        scalar_obs[0] *= self.draw_conditions["total_moves"] / self.max_total_moves
        scalar_obs[1] *= self.draw_conditions["moves_since_attack"] / self.max_moves_since_attack
        scalar_obs[2] *= self.game_phase == GamePhase.DEPLOYMENT
        scalar_obs[3] *= self.game_phase == GamePhase.MOVE

        last_selected_coord = self.p1_last_selected if self.player == 1 else self.p2_last_selected
        last_selected_obs = np.zeros((1, self.height, self.width))
        if last_selected_coord is not None:
            last_selected_obs[0][last_selected_coord] = 1

        return np.concatenate((lakes, private_obs, opp_public_obs, public_obs, moves_obs, scalar_obs, last_selected_obs))
        
    def generate_env_state(self):
        obs = self.generate_observation()
        if self.game_phase == GamePhase.DEPLOYMENT:
            action_mask = self.valid_spots_to_place()
        elif self.game_phase == GamePhase.SELECT:
            action_mask = self.valid_pieces_to_select()
        else:
            action_mask = self.valid_destinations()
        self.action_space.set_mask(action_mask.astype(bool))
        return {"obs": obs, "action_mask": action_mask}
    
    def get_public_obs(self, public_obs_info, unrevealed, pieces, movable_pieces):
        if np.sum(unrevealed[pieces]) == 0:
            probs_unmoved = np.zeros_like(unrevealed[pieces])
        else:
            probs_unmoved = unrevealed[pieces] / np.sum(unrevealed[pieces])

        if np.sum(unrevealed[movable_pieces]) == 0:
            probs_moved = np.zeros_like(unrevealed[pieces])
        else:
            probs_moved = unrevealed[pieces] / np.sum(unrevealed[movable_pieces])
        probs_moved *= np.isin(pieces, movable_pieces).astype(np.int32)

        public_obs_unmoved = public_obs_info[0] * probs_unmoved[:, None, None]
        public_obs_moved = public_obs_info[1] * probs_moved[:, None, None]
        public_obs_revealed = np.int32((public_obs_info[2] == pieces[:, None, None]))

        return public_obs_unmoved + public_obs_moved + public_obs_revealed

    def encode_move(self, action: np.ndarray):
        selected_piece = np.sum(action[0] * self.board)
        destination = np.sum(action[1] * self.board)
        if destination == Piece.EMPTY:
            return action[1] - action[0]
        else:
            return action[1] - (2 + (selected_piece - 1) / 12) * action[0]

    def get_info(self):
        """
        The get_info method returns a dictionary containing the following information: \n
        - The current player (cur_player)
        - The current board state (cur_board)
        - The shape of the board (board_shape)
        - The number of pieces in the game (num_pieces)
        - The total number of moves made (total_moves)
        - The number of moves since the last attack (moves_since_attack)
        - The current game phase (game_phase)
        - The last selected piece (last_selected). This is only valid if the game phase is MOVEMENT_PHASE,
          and it corresponds to the last piece selected by the current player.
        """
        board = np.copy(self.board)
        if self.player == Player.BLUE:
            board = np.rot90(board, 2) * -1
        return {"cur_player": np.array(self.player.value), "cur_board": board, "pieces": self.pieces,
                "board_shape": self.board.shape, "num_pieces": len(self.pieces),
                "total_moves": self.draw_conditions["total_moves"],
                "moves_since_attack": self.draw_conditions["moves_since_attack"],
                "game_phase": np.array(self.game_phase.value),
                "last_selected": None if self.game_phase != GamePhase.MOVE else
                self.p1_last_selected if self.player == Player.RED else self.p2_last_selected}
    
    def step(self, action: tuple):
        # Convert ndarray action to tuple if necessary
        if isinstance(action, np.ndarray):
            action = tuple(action.squeeze())

        valid, msg = self.validate_coord(action)
        if not valid:
            raise ValueError(msg)

        if self.game_phase == GamePhase.DEPLOYMENT:
            if self.valid_spots_to_place()[action] == 0:
                action = tuple(self.action_space.sample())
                # raise ValueError("Invalid Deployment Location")

            if self.player == Player.RED:
                self.board[action] = np.repeat(self.p1_pieces, self.p1_pieces)[self.p1_deploy_idx]
                self.p1_deploy_idx += 1
            else:
                self.board[action] = np.repeat(self.p2_pieces, self.p2_pieces)[self.p2_deploy_idx]
                self.p2_deploy_idx += 1

            if self.p2_deploy_idx == len(self.pieces):
                self.game_phase = GamePhase.SELECT

            self.board = np.rot90(self.board, 2) * -1
            self.player *= -1

            return self.generate_env_state(), 0, False, False, self.get_info()

        elif self.game_phase == GamePhase.SELECT:
            if self.valid_pieces_to_select()[action] == 0:
                action = tuple(self.action_space.sample())
                # raise ValueError("Invalid Piece Selection")

            if self.player == Player.RED:
                self.p1_last_selected = action
            else:
                self.p2_last_selected = action

            self.game_phase = GamePhase.MOVE
            return self.generate_env_state(), 0, False, False, self.get_info()

        else:
            return self.movement_step(action)
