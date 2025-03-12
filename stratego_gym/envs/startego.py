from enum import Enum, auto
import random

import numpy as np
from gymnasium import Env, spaces
import pygame

from stratego_gym.envs.config import StrategoConfig, GameMode
from stratego_gym.envs.masked_multi_descrete import MaskedMultiDiscrete
from stratego_gym.envs.primitives import Piece, Pos


class GamePhase(Enum):
    TERMINAL = auto()
    DEPLOY = auto()
    SELECT = auto()
    MOVE = auto()


class Player(Enum):
    RED = 1
    BLUE = -1

# PyGame Rendering Constants
WINDOW_SIZE = 800

def get_random_choice(valid_items):
    if np.sum(valid_items) != 0:
        probs = (valid_items / valid_items.sum()).flatten()
        flat_index = random.choices(range(len(probs)), weights=probs, k=1)[0]
        return np.unravel_index(flat_index, valid_items.shape)
    else:
        return -1, -1


class PlayerStateHandler:
    def __init__(self, player: Player):
        self.player = player
        self.pieces = None
        self.movable_pieces = None
        self.deploy_idx = 0
        self.deploy_mask = None
        self.public_obs_info = None
        self.unrevealed = None
        self.observed_moves = None
        self.last_selected = None

    def generate_state(
        self,
        pieces_num,
        deploy_mask,
        observed_history_entries: int,
        height: int,
        width: int,
    ):
        self.pieces = np.arange(len(pieces_num))[pieces_num != 0]
        self.movable_pieces = self.pieces[~np.isin(self.pieces, [Piece.FLAG, Piece.BOMB])]
        self.deploy_idx = 0
        self.deploy_mask = np.copy(deploy_mask)
        # 1st channel is unmoved, 2nd channel is moved, 3rd channel is revealed
        self.public_obs_info = np.zeros((3, height, width))
        # TODO: check whether deploy_mask or deploy_mask_opponent is needed
        self.public_obs_info[0, self.deploy_mask] = 1
        self.unrevealed = np.copy(pieces_num)
        self.observed_moves = np.zeros((observed_history_entries, height, width))


class StrategoEnv(Env):
    metadata = {"render_modes": [None, "human"]}

    def __init__(self, config: StrategoConfig | None = None, render_mode: str | None = None):
        
        self.render_mode = render_mode
        if render_mode not in [None, "human", "rgb_array"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        
        self.window = None

        self.config = config
        if config is None:
            self.config = StrategoConfig.from_game_mode(GameMode.ORIGINAL)

        self.height = self.config.height
        self.width = self.config.width

        self.game_phase = GamePhase.TERMINAL
        self.player = Player.RED
        self.board = None
        self.lakes = None

        self.p1 = PlayerStateHandler(Player.RED)
        self.p2 = PlayerStateHandler(Player.BLUE)

        self.total_moves_limit: int | None = None
        self.moves_since_attack_limit: int | None = None
        self.observed_history_entries: int | None = None

        self.draw_conditions = {"total_moves": 0, "moves_since_attack": 0}

        self.observation_channels = None
        self.observation_space = self._get_observation_space()
        self.action_space: MaskedMultiDiscrete = self._get_action_space()
        
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
            return MaskedMultiDiscrete((self.height, self.width), dtype=np.int64)
        else:
            return self.action_space
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.window = None

        if getattr(self, "height", None) != self.config.height or getattr(self, "width", None) != self.config.width:
            del self.action_space
            self.height = self.config.height
            self.width = self.config.width
            self.action_space = self._get_action_space()

        self.game_phase = GamePhase.DEPLOY
        self.player = Player.RED

        self.total_moves_limit = self.config.total_moves_limit
        self.moves_since_attack_limit = self.config.moves_since_attack_limit
        self.draw_conditions = {"total_moves": 0, "moves_since_attack": 0}

        self.observed_history_entries = self.config.observed_history_entries
        self.observation_channels = ((self.config.p1_pieces_num != 0) | (self.config.p2_pieces_num != 0)).sum()
        
        self.generate_board()
        return self.generate_env_state(), self.get_info()
    
    def generate_board(self):
        self.board = np.zeros((self.height, self.width), dtype=np.int64)
        self.lakes = np.copy(self.config.lakes_mask)
        self.board[self.lakes] = Piece.LAKE.value

        self.p1.generate_state(self.config.p1_pieces_num, self.config.p1_deploy_mask, 
                                     self.observed_history_entries, self.height, self.width)
        self.p2.generate_state(self.config.p2_pieces_num, self.config.p2_deploy_mask, 
                                     self.observed_history_entries, self.height, self.width)

    def generate_observation(self):
        lakes = self.lakes[None, :]
        private_obs = np.eye(self.observation_channels)[np.where(self.board > Piece.LAKE.value, self.board - 2, 0)].transpose(2, 0, 1)

        if self.game_phase == GamePhase.DEPLOY:
            public_obs = np.zeros((self.observation_channels, self.height, self.width))
            opp_public_obs = np.zeros((self.observation_channels, self.height, self.width))
            moves_obs = np.zeros_like(self.p1.observed_moves)
        else:
            public_obs1 = self.get_public_obs(self.p1.public_obs_info, self.p1.unrevealed, self.p1.pieces, self.p1.movable_pieces)
            public_obs2 = self.get_public_obs(self.p2.public_obs_info, self.p2.unrevealed, self.p2.pieces, self.p2.movable_pieces)
            public_obs, opp_public_obs = (public_obs1, public_obs2) if self.player == Player.RED else (public_obs2, public_obs1)
            moves_obs = self.p1.observed_moves if self.player == Player.RED else self.p2.observed_moves

        scalar_obs = np.ones((4, self.height, self.width))
        scalar_obs[0] *= self.draw_conditions["total_moves"] / self.total_moves_limit
        scalar_obs[1] *= self.draw_conditions["moves_since_attack"] / self.moves_since_attack_limit
        scalar_obs[2] *= self.game_phase == GamePhase.DEPLOY
        scalar_obs[3] *= self.game_phase == GamePhase.MOVE

        last_selected_coord = self.p1.last_selected if self.player == Player.RED else self.p2.last_selected
        last_selected_obs = np.zeros((1, self.height, self.width))
        if last_selected_coord is not None:
            last_selected_obs[0][last_selected_coord] = 1

        return np.concatenate((lakes, private_obs, opp_public_obs, public_obs, moves_obs, scalar_obs, last_selected_obs))
        
    def generate_env_state(self):
        obs = self.generate_observation()
        if self.game_phase == GamePhase.DEPLOY:
            action_mask = self.valid_spots_to_place()
        elif self.game_phase == GamePhase.SELECT:
            action_mask = self.valid_pieces_to_select()
        else:
            action_mask = self.valid_destinations()
        self.action_space.set_mask(action_mask.astype(bool))
        print(action_mask)
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
        return {"cur_player": np.array(self.player.value), "cur_board": board, #"pieces": self.pieces,
                #"board_shape": self.board.shape, "num_pieces": len(self.pieces),
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

        if self.game_phase == GamePhase.DEPLOY:
            if self.valid_spots_to_place()[action] == 0:
                action = tuple(self.action_space.sample())
                # raise ValueError("Invalid Deployment Location")

            if self.player == Player.RED:
                self.board[action] = np.repeat(self.p1.pieces, self.p1.pieces)[self.p1.deploy_idx]
                self.p1.deploy_idx += 1
            else:
                self.board[action] = np.repeat(self.p2.pieces, self.p2.pieces)[self.p2.deploy_idx]
                self.p2.deploy_idx += 1

            if self.p2.deploy_idx == len(self.p2.pieces):
                self.game_phase = GamePhase.SELECT

            self.board = np.rot90(self.board, 2) * -1
            self.player = Player(self.player.value * -1)

            return self.generate_env_state(), 0, False, False, self.get_info()

        elif self.game_phase == GamePhase.SELECT:
            if self.valid_pieces_to_select()[action] == 0:
                action = tuple(self.action_space.sample())
                # raise ValueError("Invalid Piece Selection")

            if self.player == Player.RED:
                self.p1_last_selected = action
            else:
                self.p2_last_selected = action

            self.game_phase = GamePhase.MOVEMENT
            return self.generate_env_state(), 0, False, False, self.get_info()

        else:
            return self.movement_step(action)
        
    def movement_step(self, action: tuple):
        source = self.p1_last_selected if self.player == 1 else self.p2_last_selected
        dest = action

        # Action is a tuple representing a coordinate on the board
        valid, msg = self.check_action_valid(source, dest)
        if not valid:
            action = tuple(self.action_space.sample())
            dest = action
            # raise ValueError(msg)

        # Get Selected Piece Identity and Destination Identity
        selected_piece = self.board[source]
        destination = self.board[dest]

        action = np.zeros((2,) + self.board.shape, dtype=np.int64)
        action[0][source] = 1
        action[1][dest] = 1

        # Initialize Reward, Termination, and Info
        reward = 0
        terminated = False

        # Check if draw conditions are met
        if self.draw_conditions["total_moves"] >= 2000 or self.draw_conditions["moves_since_attack"] >= 200:
            self.board = np.rot90(self.board, 2) * -1
            self.player *= -1
            self.game_phase = GamePhase.TERMINAL
            return self.generate_env_state(), 0, True, False, self.get_info()

        # Update Draw conditions
        self.draw_conditions["total_moves"] += 1
        if destination == Piece.EMPTY:
            self.draw_conditions["moves_since_attack"] += 1
        else:
            self.draw_conditions["moves_since_attack"] = 0

        # Update Move Histories
        self.p1.observed_moves = np.roll(self.p1.observed_moves, 1, axis=0)
        self.p2.observed_moves = np.roll(self.p2.observed_moves, 1, axis=0)
        cur_player_moves = self.p1.observed_moves if self.player == 1 else self.p2.observed_moves
        other_player_moves = self.p2.observed_moves if self.player == 1 else self.p1.observed_moves

        move = self.encode_move(action)
        cur_player_moves[0] = move
        other_player_moves[0] = np.rot90(move, 2) * -1

        # Perform Move Logic
        if self.player == Player.RED:
            cur_player_public_info = self.p1.public_obs_info if self.player == 1 else self.p2.public_obs_info
            cur_player_unrevealed = self.p1.unrevealed if self.player == 1 else self.p2.unrevealed
            other_player_public_info = self.p2.public_obs_info if self.player == 1 else self.p1.public_obs_info
            other_player_unrevealed = self.p2.unrevealed if self.player == 1 else self.p1.unrevealed
        else:
            cur_player_public_info = self.p2.public_obs_info
            cur_player_unrevealed = self.p2.unrevealed
            other_player_public_info = self.p1.public_obs_info
            other_player_unrevealed = self.p1.unrevealed

        if ((selected_piece != Piece.MINER and destination == -Piece.BOMB) or  # Bomb
                (selected_piece == -destination)):  # Equal Strength
            # Remove Both Pieces
            self.board *= np.prod(1 - action, axis=0)
            cur_player_public_info *= np.prod(1 - action, axis=0)
            other_player_public_info *= np.rot90(np.prod(1 - action, axis=0), 2)
            cur_player_unrevealed[selected_piece] -= 1
            other_player_unrevealed[destination] -= 1
        elif ((selected_piece == Piece.SPY and destination == -Piece.MARSHAL) or  # Spy vs Marshal
              (selected_piece > -destination) or  # Attacker is stronger (Bomb case already handled)
              (destination == -Piece.FLAG)):  # Enemy Flag Found
            # Remove Enemy Piece
            self.board *= np.prod(1 - action, axis=0)
            self.board += action[1] * selected_piece
            cur_player_public_info *= 1 - action[0]
            if destination != Piece.EMPTY:
                cur_player_public_info[2] += action[1] * selected_piece
                other_player_public_info *= np.rot90(1 - action[1], 2)
                cur_player_unrevealed[selected_piece] -= 1
                other_player_unrevealed[destination] -= 1
            else:
                scout_move = np.sum(np.abs(np.argwhere(action[0] == 1)[0] - np.argwhere(action[1] == 1)[0])) > 1
                if scout_move:
                    cur_player_public_info[2] += action[1] * selected_piece
                    cur_player_unrevealed[selected_piece] -= 1
                else:
                    cur_player_public_info[1] += action[1]

            if destination == -Piece.FLAG:
                reward = 1
                terminated = True
        elif selected_piece < -destination:
            # Remove Attacker
            self.board *= 1 - action[0]
            cur_player_public_info *= 1 - action[0]
            other_player_public_info *= np.rot90(1 - action[1], 2)
            other_player_public_info[2] += np.rot90(action[1] * destination, 2)
            cur_player_unrevealed[selected_piece] -= 1
            other_player_unrevealed[destination] -= 1

        self.board = np.rot90(self.board, 2) * -1
        self.player *= -1

        # Check if any pieces can be moved. If one player has no movable pieces, the other player wins.
        # If both players have no movable pieces, the game is a draw.
        if not terminated and (np.sum(self.board >= Piece.SPY) == 0 or self.valid_pieces_to_select().sum() == 0):
            draw_game = (np.sum(self.board <= -Piece.SPY) == 0) or self.valid_pieces_to_select(is_other_player=True).sum() == 0
            self.game_phase = GamePhase.TERMINAL
            return self.generate_env_state(), 0 if draw_game else 1, True, False, self.get_info()

        self.game_phase = GamePhase.TERMINAL if terminated else GamePhase.SELECT
        return self.generate_env_state(), reward, terminated, False, self.get_info()

    def validate_coord(self, coord):
        if len(coord) != 2 and all(isinstance(item, int) for item in coord):
            return False, "Source tuple size or type is not as expected"

        if coord[0] < 0 or coord[0] >= self.board.shape[0]:
            return False, "Source row is out of bounds"

        if coord[1] < 0 or coord[1] >= self.board.shape[1]:
            return False, "Source column is out of bounds"

        return True, None

    def check_action_valid(self, src: tuple, dest: tuple):
        valid, msg = self.validate_coord(src)
        if not valid:
            return False, msg

        valid, msg = self.validate_coord(dest)
        if not valid:
            return False, msg

        selected_piece = self.board[src]
        if selected_piece < Piece.SPY:
            return False, "Selected piece cannot be moved by player"

        destination = self.board[dest]

        if abs(destination) == Piece.LAKE:
            return False, "Destination is an obstacle"

        if destination > Piece.LAKE:
            return False, "Destination is already occupied by player's piece"

        if selected_piece != Piece.SCOUT:
            selected_piece_coord = np.array(src)
            destination_coord = np.array(dest)
            if np.sum(np.abs(selected_piece_coord - destination_coord)) != 1:
                return False, "Invalid move"
        else:
            selected_piece_coord = np.array(src)
            destination_coord = np.array(dest)

            if selected_piece_coord[0] != destination_coord[0] and selected_piece_coord[1] != destination_coord[1]:
                return False, "Scouts can only move in straight lines"

            path_slice = self.board[
                         min(selected_piece_coord[0], destination_coord[0]) + 1:max(selected_piece_coord[0],
                                                                                    destination_coord[0]),
                         min(selected_piece_coord[1], destination_coord[1]) + 1:max(selected_piece_coord[1],
                                                                                    destination_coord[1])]

            if np.any(path_slice != 0):
                return False, "Pieces in the path of scout"

        return True, "Valid Action"

    def valid_spots_to_place(self) -> np.ndarray:
        deploy_mask = self.p1.deploy_mask if self.player == Player.RED else np.rot90(self.p2.deploy_mask, 2)
        print(self.board)
        return (self.board == Piece.EMPTY.value) & deploy_mask

    def valid_pieces_to_select(self, is_other_player=False) -> np.ndarray:
        padded_board = np.pad(self.board, 1, constant_values=Piece.LAKE)
        padded_board[padded_board == -Piece.LAKE] = Piece.LAKE

        # Shift the padded array in all four directions
        shift_left = np.roll(padded_board, 1, axis=1)[1:-1, 1:-1]
        shift_right = np.roll(padded_board, -1, axis=1)[1:-1, 1:-1]
        shift_up = np.roll(padded_board, 1, axis=0)[1:-1, 1:-1]
        shift_down = np.roll(padded_board, -1, axis=0)[1:-1, 1:-1]

        # Check conditions to create the boolean array
        surrounded = (shift_left >= Piece.LAKE) & (shift_right >= Piece.LAKE) & (shift_up >= Piece.LAKE) & (
                shift_down >= Piece.LAKE)

        return np.logical_and((self.board <= -Piece.SPY) if is_other_player else (self.board >= Piece.SPY), ~surrounded).astype(int)

    def valid_destinations(self):
        if self.game_phase != GamePhase.MOVE:
            return np.zeros_like(self.board)

        selected = self.p1_last_selected if self.player == 1 else self.p2_last_selected
        selected_piece_val = self.board[selected]
        board_shape = np.array(self.board.shape)

        directions = np.array([[0, 0, 1, -1], [1, -1, 0, 0]])
        destinations = np.zeros_like(self.board)

        if selected_piece_val == Piece.SCOUT:
            for direction in directions.T:
                positions = np.array(selected)[:, None] + direction[:, None]
                encountered_enemy = 0
                while (
                        np.all(positions >= 0, axis=0)
                        and np.all(positions < board_shape[:, None], axis=0)
                        and encountered_enemy < 1
                ):
                    if self.board[positions[0], positions[1]] != Piece.EMPTY:
                        if self.board[positions[0], positions[1]] > -Piece.FLAG:
                            break
                        encountered_enemy += 1
                    destinations[positions[0], positions[1]] = 1
                    positions += direction[:, None]
            return destinations

        else:
            positions = np.array(selected)[:, None] + directions
            valid_positions = positions[
                              :,
                              (np.all(positions >= 0, axis=0)) &
                              (np.all(positions < board_shape[:, None], axis=0))
                              ]
            mask = (
                    (self.board[valid_positions[0], valid_positions[1]] <= Piece.EMPTY) &
                    (self.board[valid_positions[0], valid_positions[1]] != -Piece.LAKE)
            )
            valid_positions = valid_positions[:, mask]
            destinations[valid_positions[0], valid_positions[1]] = 1
            return destinations

    def get_random_action(self) -> tuple:
        if self.game_phase == GamePhase.DEPLOY:
            valid_spots = self.valid_spots_to_place()
            return get_random_choice(valid_spots)
        elif self.game_phase == GamePhase.SELECT:
            pieces_to_select = self.valid_pieces_to_select()
            return get_random_choice(pieces_to_select)
        elif self.game_phase == GamePhase.MOVE:
            destinations = self.valid_destinations()
            return get_random_choice(destinations)
        else:
            return -1, -1
        
    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (WINDOW_SIZE, WINDOW_SIZE)
            )

        cell_size = (WINDOW_SIZE // self.board.shape[0], WINDOW_SIZE // self.board.shape[1])

        # Calculate the number of cells in each direction
        num_cells_r = self.board.shape[0]
        num_cells_c = self.board.shape[1]

        board = np.copy(self.board)
        if self.player == -1:
            board = np.rot90(board, 2) * -1

        font = pygame.font.Font(None, 80)
        # Draw the grid
        for r in range(num_cells_r):
            for c in range(num_cells_c):
                rect = pygame.Rect(c * cell_size[0], r * cell_size[1], cell_size[0], cell_size[1])
                rect_center = rect.center
                if abs(board[r][c]) == Piece.LAKE:
                    pygame.draw.rect(self.window, (158, 194, 230), rect)
                    text = None
                elif board[r][c] >= Piece.FLAG:
                    pygame.draw.rect(self.window, (217, 55, 58), rect)
                    render_text = 'F' if board[r][c] == Piece.FLAG else 'B' if board[r][c] == Piece.BOMB else str(board[r][c] - 3)
                    text = font.render(render_text, True, (255, 255, 255))
                elif board[r][c] <= -Piece.FLAG:
                    pygame.draw.rect(self.window, (24, 118, 181), rect)
                    render_text = 'F' if board[r][c] == -Piece.FLAG else 'B' if board[r][c] == -Piece.BOMB else str(-(board[r][c] + 3))
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
