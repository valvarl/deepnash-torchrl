from enum import Enum, auto
import random

import numpy as np
from gymnasium import spaces

from stratego.core.config import StrategoConfig, GameMode
from stratego.core.detectors import TwoSquareDetector, ChasingDetector
from stratego.core.masked_multi_descrete import MaskedMultiDiscrete
from stratego.core.primitives import Piece, Player
from stratego.core.stratego_base import StrategoEnvBase


class GamePhase(Enum):
    TERMINAL = 0
    DEPLOY = 1
    SELECT = 2
    MOVE = 3


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
        self.last_selected_piece = None

    def generate_state(
        self,
        pieces_num,
        deploy_mask,
        observed_history_entries: int,
        height: int,
        width: int,
    ):
        self.pieces = np.arange(len(pieces_num))[pieces_num != 0]
        self.movable_pieces = self.pieces[
            ~np.isin(self.pieces, [Piece.FLAG.value, Piece.BOMB.value])
        ]
        # print(self.pieces)
        # print(self.movable_pieces)
        self.deploy_idx = 0
        self.deploy_mask = np.copy(deploy_mask)
        # 1st channel is unmoved, 2nd channel is moved, 3rd channel is revealed
        self.public_obs_info = np.zeros((3, height, width))
        self.unrevealed = np.copy(pieces_num)
        # print(self.unrevealed)
        self.observed_moves = np.zeros((observed_history_entries, height, width))


class StrategoEnv(StrategoEnvBase):
    def __init__(
        self, config: StrategoConfig | None = None, render_mode: str | None = None
    ):
        super().__init__(render_mode=render_mode)

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
        self.two_square_detector = TwoSquareDetector()
        self.chasing_detector = ChasingDetector()

        self.total_moves_limit: int | None = None
        self.moves_since_attack_limit: int | None = None
        self.observed_history_entries: int | None = None

        self.draw_conditions = {"total_moves": 0, "moves_since_attack": 0}

        self.allowed_pieces = None
        self.observation_space = self._get_observation_space()
        self.action_space: MaskedMultiDiscrete = self._get_action_space()

    def _get_observation_space(self):
        observation_channels = (
            len(self.config.allowed_pieces) * 3
            + self.config.observed_history_entries
            + 6
        )
        shape = (observation_channels, self.config.height, self.config.width)
        mask_shape = (self.config.height, self.config.width)

        return spaces.Dict(
            {
                "obs": spaces.Box(low=-3, high=1, shape=shape, dtype=np.float64),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=mask_shape, dtype=np.int64
                ),
            }
        )

    def _get_action_space(self):
        if getattr(self, "action_space", None) is None:
            return MaskedMultiDiscrete((self.height, self.width), dtype=np.int64)
        else:
            return self.action_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        if (
            getattr(self, "height", None) != self.config.height
            or getattr(self, "width", None) != self.config.width
        ):
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
        self.allowed_pieces = self.config.allowed_pieces

        self.board = np.zeros((self.height, self.width), dtype=np.int64)
        self.lakes = np.copy(self.config.lakes_mask)
        self.board[self.lakes] = Piece.LAKE.value

        self.p1.generate_state(
            self.config.p1_pieces_num,
            self.config.p1_deploy_mask,
            self.observed_history_entries,
            self.height,
            self.width,
        )
        self.p2.generate_state(
            self.config.p2_pieces_num,
            self.config.p2_deploy_mask,
            self.observed_history_entries,
            self.height,
            self.width,
        )

        self.two_square_detector.reset()
        self.chasing_detector.reset()

        return self.generate_env_state(), self.get_info()

    def generate_observation(self):
        lakes = self.lakes[None, :]
        private_obs = np.eye(len(Piece))[
            np.where(self.board > Piece.LAKE.value, self.board, 0)
        ].transpose(2, 0, 1)[self.allowed_pieces]

        if self.game_phase == GamePhase.DEPLOY:
            public_obs = np.zeros((len(self.allowed_pieces), self.height, self.width))
            opp_public_obs = np.zeros(
                (len(self.allowed_pieces), self.height, self.width)
            )
            moves_obs = np.zeros_like(self.p1.observed_moves)
        else:
            public_obs1 = self.get_public_obs(
                self.p1.public_obs_info,
                self.p1.unrevealed,
                self.p1.pieces,
                self.p1.movable_pieces,
            )
            public_obs2 = self.get_public_obs(
                self.p2.public_obs_info,
                self.p2.unrevealed,
                self.p2.pieces,
                self.p2.movable_pieces,
            )
            public_obs, opp_public_obs = (
                (public_obs1, public_obs2)
                if self.player == Player.RED
                else (public_obs2, public_obs1)
            )
            moves_obs = (
                self.p1.observed_moves
                if self.player == Player.RED
                else self.p2.observed_moves
            )

        scalar_obs = np.ones((4, self.height, self.width))
        scalar_obs[0] *= self.draw_conditions["total_moves"] / self.total_moves_limit
        scalar_obs[1] *= (
            self.draw_conditions["moves_since_attack"] / self.moves_since_attack_limit
        )
        scalar_obs[2] *= self.game_phase == GamePhase.DEPLOY
        scalar_obs[3] *= self.game_phase == GamePhase.MOVE

        last_selected_coord = (
            self.p1.last_selected
            if self.player == Player.RED
            else self.p2.last_selected
        )
        last_selected_obs = np.zeros((1, self.height, self.width))
        if last_selected_coord is not None:
            last_selected_obs[0][last_selected_coord] = 1

        return np.concatenate(
            (
                lakes,
                private_obs,
                opp_public_obs,
                public_obs,
                moves_obs,
                scalar_obs,
                last_selected_obs,
            )
        )

    def generate_env_state(self):
        obs = self.generate_observation()
        if self.game_phase == GamePhase.DEPLOY:
            action_mask = self.valid_spots_to_place()
        elif self.game_phase == GamePhase.SELECT:
            action_mask = self.valid_pieces_to_select()
            # print(self.chasing_detector.chase_moves)
        else:
            action_mask = self.valid_destinations()
            # print('MOVE\n', action_mask)
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
        if destination == Piece.EMPTY.value:
            return action[1] - action[0]
        else:
            return action[1] - (2 + (selected_piece - 3) / 12) * action[0]

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
        return {
            "cur_player": np.array(self.player.value),
            "cur_board": board,
            "pieces": self.allowed_pieces,
            "board_shape": self.board.shape,
            "num_pieces": len(self.allowed_pieces),
            "total_moves": self.draw_conditions["total_moves"],
            "moves_since_attack": self.draw_conditions["moves_since_attack"],
            "game_phase": np.array(self.game_phase.value),
            "last_selected": (
                None
                if self.game_phase != GamePhase.MOVE
                else (
                    self.p1.last_selected
                    if self.player == Player.RED
                    else self.p2.last_selected
                )
            ),
        }

    def step(self, action):
        # 1. Преобразуем action в tuple при необходимости
        if isinstance(action, np.ndarray):
            action = tuple(action.squeeze())

        # 2. Проверяем, что координаты валидны
        valid, msg = self.validate_coord(action)
        if not valid:
            raise ValueError(msg)

        # 3. Переходим к нужной фазе игры
        if self.game_phase == GamePhase.DEPLOY:
            return self._deploy_step(action)
        elif self.game_phase == GamePhase.SELECT:
            return self._select_step(action)
        else:
            # Фаза MOVE (или другие фазы) обрабатываются в movement_step
            return self.movement_step(action)

    def _deploy_step(self, action):
        # Если место для размещения недоступно – выбираем случайное
        if self.valid_spots_to_place()[action] == 0:
            action = tuple(self.action_space.sample())

        # Выбираем состояние текущего и противоположного игрока
        player_state = self.p1 if self.player == Player.RED else self.p2
        opp_state = self.p2 if self.player == Player.RED else self.p1

        # Размещаем фигуру (повторяющаяся логика вынесена в цикл)
        self.board[action] = np.repeat(np.arange(len(Piece)), player_state.unrevealed)[
            player_state.deploy_idx
        ]
        player_state.deploy_idx += 1

        # Если у оппонента все фигуры уже размещены, а у текущего игрока – ещё нет
        if (
            opp_state.deploy_idx == opp_state.unrevealed.sum()
            and player_state.deploy_idx != player_state.unrevealed.sum()
        ):
            return self.generate_env_state(), 0, False, False, self.get_info()

        # Если оба игрока закончили расстановку
        if (
            self.p1.deploy_idx == self.p1.unrevealed.sum()
            and self.p2.deploy_idx == self.p2.unrevealed.sum()
        ):
            # Проверяем, не закончилась ли игра раньше времени
            if (
                np.sum(self.board <= -Piece.SPY.value) == 0
                or self.valid_pieces_to_select(is_other_player=True).sum() == 0
            ):
                # Ничья, если у второго игрока тоже нет ходов/фигур
                draw_game = (
                    np.sum(self.board >= Piece.SPY.value) == 0
                    or self.valid_pieces_to_select().sum() == 0
                )
                self.game_phase = GamePhase.TERMINAL
                return (
                    self.generate_env_state(),
                    (0 if draw_game else 1),
                    True,
                    False,
                    self.get_info(),
                )

            # Если игра не закончена – переходим к фазе SELECT
            self.game_phase = GamePhase.SELECT
            self._update_public_obs_info()

        # Вращаем доску и меняем игрока в конце фазы DEPLOY
        self._rotate_board_and_switch_player()
        return self.generate_env_state(), 0, False, False, self.get_info()

    def _select_step(self, action):
        # Проверяем, что выбранная фигура валидна
        if self.valid_pieces_to_select()[action] == 0:
            action = tuple(self.action_space.sample())

        player_state = self.p1 if self.player == Player.RED else self.p2
        player_state.last_selected = action
        player_state.last_selected_piece = Piece(self.board[action])

        # Переходим к фазе MOVE
        self.game_phase = GamePhase.MOVE
        return self.generate_env_state(), 0, False, False, self.get_info()

    def _update_public_obs_info(self):
        """
        Обновляет публичную информацию игроков:
        • Если ходит RED, то у RED отмечается собственная часть доски,
          у BLUE – повёрнутая часть доски, и наоборот.
        """
        if self.player == Player.RED:
            self.p1.public_obs_info[0, self.board * self.p1.deploy_mask > 0] = 1
            self.p2.public_obs_info[
                0, np.rot90(self.board * self.p2.deploy_mask > 0, 2)
            ] = 1
        else:
            self.p1.public_obs_info[
                0, np.rot90(self.board * self.p1.deploy_mask > 0, 2)
            ] = 1
            self.p2.public_obs_info[0, self.board * self.p2.deploy_mask > 0] = 1

    def _rotate_board_and_switch_player(self):
        """Поворачивает доску на 180 градусов и меняет текущего игрока."""
        self.board = np.rot90(self.board, 2) * -1
        self.player = Player(-self.player.value)

    def movement_step(self, action: tuple):
        source = (
            self.p1.last_selected
            if self.player == Player.RED
            else self.p2.last_selected
        )
        dest = action

        # Action is a tuple representing a coordinate on the board
        valid, msg = self.check_action_valid(source, dest)
        if not valid:
            action = tuple(self.action_space.sample())
            dest = action
            # raise ValueError(msg)

        if self.player == Player.RED:
            self.p1.last_selected = action
        else:
            self.p2.last_selected = action

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
        if (
            self.draw_conditions["total_moves"] >= self.total_moves_limit
            or self.draw_conditions["moves_since_attack"]
            >= self.moves_since_attack_limit
        ):
            self.board = np.rot90(self.board, 2) * -1
            self.player = Player(self.player.value * -1)
            self.game_phase = GamePhase.TERMINAL
            return self.generate_env_state(), 0, True, False, self.get_info()

        # Update Draw conditions
        self.draw_conditions["total_moves"] += 1
        if destination == Piece.EMPTY.value:
            self.draw_conditions["moves_since_attack"] += 1
        else:
            self.draw_conditions["moves_since_attack"] = 0

        # Update Move Histories
        self.p1.observed_moves = np.roll(self.p1.observed_moves, 1, axis=0)
        self.p2.observed_moves = np.roll(self.p2.observed_moves, 1, axis=0)
        cur_player_moves = (
            self.p1.observed_moves
            if self.player == Player.RED
            else self.p2.observed_moves
        )
        other_player_moves = (
            self.p2.observed_moves
            if self.player == Player.RED
            else self.p1.observed_moves
        )

        move = self.encode_move(action)
        cur_player_moves[0] = move
        other_player_moves[0] = np.rot90(move, 2) * -1

        self.two_square_detector.update(
            self.player, Piece(selected_piece), source, dest
        )
        _source, _dest = source, dest
        if self.player == Player.BLUE:
            _source = (self.height - source[0] - 1, self.width - source[1] - 1)
            _dest = (self.height - dest[0] - 1, self.width - dest[1] - 1)
        self.chasing_detector.update(
            self.player, Piece(selected_piece), _source, _dest, self.board
        )

        # Perform Move Logic
        cur_player_public_info = (
            self.p1.public_obs_info
            if self.player == Player.RED
            else self.p2.public_obs_info
        )
        cur_player_unrevealed = (
            self.p1.unrevealed if self.player == Player.RED else self.p2.unrevealed
        )
        other_player_public_info = (
            self.p2.public_obs_info
            if self.player == Player.RED
            else self.p1.public_obs_info
        )
        other_player_unrevealed = (
            self.p2.unrevealed if self.player == Player.RED else self.p1.unrevealed
        )

        if selected_piece == -destination:  # Equal Strength
            # Remove Both Pieces
            self.board *= np.prod(1 - action, axis=0)
            cur_player_public_info *= np.prod(1 - action, axis=0)
            other_player_public_info *= np.rot90(np.prod(1 - action, axis=0), 2)
            cur_player_unrevealed[selected_piece] -= 1
            other_player_unrevealed[destination] -= 1
        elif (
            (
                selected_piece == Piece.SPY.value
                and destination == -Piece.MARSHAL.value
            )  # Spy vs Marshal
            or (
                selected_piece > -destination
                and (
                    selected_piece == Piece.MINER.value
                    and destination == -Piece.BOMB.value
                    or destination != -Piece.BOMB.value
                )
            )  # Attacker is stronger (Bomb case already handled)
            or (destination == -Piece.FLAG.value)
        ):  # Enemy Flag Found
            # Remove Enemy Piece
            self.board *= np.prod(1 - action, axis=0)
            self.board += action[1] * selected_piece
            cur_player_public_info *= 1 - action[0]
            if destination != Piece.EMPTY.value:
                cur_player_public_info[2] += action[1] * selected_piece
                other_player_public_info *= np.rot90(1 - action[1], 2)
                cur_player_unrevealed[selected_piece] -= 1
                other_player_unrevealed[destination] -= 1
            else:
                scout_move = (
                    np.sum(
                        np.abs(
                            np.argwhere(action[0] == 1)[0]
                            - np.argwhere(action[1] == 1)[0]
                        )
                    )
                    > 1
                )
                if scout_move:
                    cur_player_public_info[2] += action[1] * selected_piece
                    cur_player_unrevealed[selected_piece] -= 1
                else:
                    cur_player_public_info[1] += action[1]

            if destination == -Piece.FLAG.value:
                reward = 1
                terminated = True
        elif selected_piece < -destination or destination == -Piece.BOMB.value:
            # Remove Attacker
            self.board *= 1 - action[0]
            cur_player_public_info *= 1 - action[0]
            other_player_public_info *= np.rot90(1 - action[1], 2)
            other_player_public_info[2] += np.rot90(action[1] * destination, 2)
            cur_player_unrevealed[selected_piece] -= 1
            other_player_unrevealed[destination] -= 1

        self.board = np.rot90(self.board, 2) * -1
        self.player = Player(self.player.value * -1)

        # Check if any pieces can be moved. If one player has no movable pieces, the other player wins.
        # If both players have no movable pieces, the game is a draw.
        if not terminated:
            current_player_no_moves = (
                np.sum(self.board >= Piece.SPY.value) == 0
                or self.valid_pieces_to_select().sum() == 0
            )
            if current_player_no_moves:
                other_player_no_moves = (
                    np.sum(self.board <= -Piece.SPY.value) == 0
                    or self.valid_pieces_to_select(is_other_player=True).sum() == 0
                )
                self.game_phase = GamePhase.TERMINAL
                return (
                    self.generate_env_state(),
                    0 if other_player_no_moves else 1,
                    True,
                    False,
                    self.get_info(),
                )

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
        if selected_piece < Piece.SPY.value:
            return False, "Selected piece cannot be moved by player"

        destination = self.board[dest]

        if abs(destination) == Piece.LAKE.value:
            return False, "Destination is an obstacle"

        if destination > Piece.LAKE.value:
            return False, "Destination is already occupied by player's piece"

        if selected_piece != Piece.SCOUT.value:
            selected_piece_coord = np.array(src)
            destination_coord = np.array(dest)
            if np.sum(np.abs(selected_piece_coord - destination_coord)) != 1:
                return False, "Invalid move"
        else:
            selected_piece_coord = np.array(src)
            destination_coord = np.array(dest)

            if (
                selected_piece_coord[0] != destination_coord[0]
                and selected_piece_coord[1] != destination_coord[1]
            ):
                return False, "Scouts can only move in straight lines"

            path_slice = self.board[
                min(selected_piece_coord[0], destination_coord[0])
                + 1 : max(selected_piece_coord[0], destination_coord[0]),
                min(selected_piece_coord[1], destination_coord[1])
                + 1 : max(selected_piece_coord[1], destination_coord[1]),
            ]

            if np.any(path_slice != 0):
                return False, "Pieces in the path of scout"

        if not self.two_square_detector.validate_move(
            self.player, Piece(selected_piece), src, dest
        ):
            return (
                False,
                f"Two-square rule violation: {self.two_square_detector.get_player(self.player)}",
            )

        _src, _dest = src, dest
        if self.player == Player.BLUE:
            _src = (self.height - src[0] - 1, self.width - src[1] - 1)
            _dest = (self.height - dest[0] - 1, self.width - dest[1] - 1)
        if not self.chasing_detector.validate_move(
            self.player, Piece(selected_piece), _src, _dest, self.board
        ):
            return (
                False,
                f"More-square rule violation: {self.chasing_detector.chase_moves}",
            )

        return True, "Valid Action"

    def valid_spots_to_place(self) -> np.ndarray:
        deploy_mask = (
            self.p1.deploy_mask
            if self.player == Player.RED
            else np.rot90(self.p2.deploy_mask, 2)
        )
        return (self.board == Piece.EMPTY.value) & deploy_mask

    def valid_pieces_to_select(
        self, is_other_player=False, debug_print=False
    ) -> np.ndarray:
        board = self.board if not is_other_player else np.rot90(self.board, 2) * -1
        padded_board = np.pad(board, 1, constant_values=Piece.LAKE.value)
        padded_board[padded_board == -Piece.LAKE.value] = Piece.LAKE.value

        # Shift the padded array in all four directions
        shift_left = np.roll(padded_board, 1, axis=1)[1:-1, 1:-1]
        shift_right = np.roll(padded_board, -1, axis=1)[1:-1, 1:-1]
        shift_up = np.roll(padded_board, 1, axis=0)[1:-1, 1:-1]
        shift_down = np.roll(padded_board, -1, axis=0)[1:-1, 1:-1]

        # Check conditions to create the boolean array
        surrounded = (
            (shift_left >= Piece.LAKE.value)
            & (shift_right >= Piece.LAKE.value)
            & (shift_up >= Piece.LAKE.value)
            & (shift_down >= Piece.LAKE.value)
        )

        # Two-square and More-square rules
        player = self.player if not is_other_player else Player(-1 * self.player.value)
        p = self.p1 if player == Player.RED else self.p2
        pos, piece = p.last_selected, p.last_selected_piece
        if pos is None:
            return np.logical_and(board >= Piece.SPY.value, ~surrounded).astype(int)

        valid_two_square, pos_twosq = self.two_square_detector.validate_select(
            player, piece, pos
        )
        _pos = (
            pos
            if player == Player.RED
            else (self.height - pos[0] - 1, self.width - pos[1] - 1)
        )
        valid_chasing, pos_chasing = self.chasing_detector.validate_select(
            player, piece, _pos, board
        )

        if debug_print:
            print("check cond", not (valid_two_square and valid_chasing))
            print(player)
            print(piece)
            print(pos)
            p = self.p2 if player == Player.RED else self.p1
            print(p.last_selected, p.last_selected_piece)
            print(self.two_square_detector.validate_select(player, piece, pos))

        if not (valid_two_square and valid_chasing):
            mask = np.zeros((self.height, self.width), dtype=bool)
            if not valid_two_square:
                start_pos, end_pos = pos_twosq
                if start_pos == end_pos:
                    mask[start_pos] = True
                else:
                    if start_pos[0] == end_pos[0]:
                        mask[
                            start_pos[0],
                            min(start_pos[1], end_pos[1]) : max(
                                start_pos[1], end_pos[1]
                            )
                            + 1,
                        ] = True
                    else:
                        mask[
                            min(start_pos[0], end_pos[0]) : max(
                                start_pos[0], end_pos[0]
                            )
                            + 1,
                            start_pos[1],
                        ] = True

            if not valid_chasing:
                for _pos in pos_chasing:
                    if player == Player.BLUE:
                        _pos = (self.height - pos[0] - 1, self.width - pos[1] - 1)
                    mask[_pos] = True

            surrounded_square = 0
            for i, j in zip([-1, 1, 0, 0], [0, 0, -1, 1]):
                _pos = (pos[0] + i, pos[1] + j)
                while 0 <= _pos[0] < self.height and 0 <= _pos[1] < self.width:
                    if not mask[_pos]:
                        if (
                            board[_pos] >= Piece.LAKE.value
                            or board[_pos] == -Piece.LAKE.value
                        ):
                            surrounded_square += 1
                        break
                    elif piece == Piece.SCOUT and board[_pos] <= -Piece.SPY.value:
                        surrounded_square += 1
                        break

                    if piece != Piece.SCOUT:
                        if mask[_pos]:
                            surrounded_square += 1
                        break
                    _pos = (_pos[0] + i, _pos[1] + j)
                else:
                    if _pos[0] < 0 or _pos[0] >= self.height:
                        surrounded_square += 1
                    elif _pos[1] < 0 or _pos[1] >= self.width:
                        surrounded_square += 1

            if surrounded_square == 4:
                surrounded[pos] = True

            if debug_print:
                print(
                    player,
                    "BOARD\n",
                    board,
                    "MASK\n",
                    mask,
                    surrounded_square,
                    "SURROUNDED\n",
                    surrounded,
                )

        return np.logical_and(board >= Piece.SPY.value, ~surrounded).astype(int)

    def valid_destinations(self):
        if self.game_phase != GamePhase.MOVE:
            return np.zeros_like(self.board)

        selected = (
            self.p1.last_selected
            if self.player == Player.RED
            else self.p2.last_selected
        )
        selected_piece_val = self.board[selected]
        board_shape = np.array(self.board.shape)

        directions = np.array([[0, 0, 1, -1], [1, -1, 0, 0]])
        destinations = np.zeros_like(self.board)

        two_square_mask = None
        valid, positions = self.two_square_detector.validate_select(
            self.player, Piece(selected_piece_val), selected
        )
        if not valid:
            two_square_mask = np.ones((self.height, self.width), dtype=bool)
            start_pos, end_pos = positions
            if start_pos == end_pos:
                two_square_mask[start_pos] = False
            else:
                if start_pos[0] == end_pos[0]:
                    two_square_mask[
                        start_pos[0],
                        min(start_pos[1], end_pos[1]) : max(start_pos[1], end_pos[1])
                        + 1,
                    ] = False
                else:
                    two_square_mask[
                        min(start_pos[0], end_pos[0]) : max(start_pos[0], end_pos[0])
                        + 1,
                        start_pos[1],
                    ] = False

        chasing_mask = None
        _selected = selected
        if self.player == Player.BLUE:
            _selected = (self.height - selected[0] - 1, self.width - selected[1] - 1)
        valid, position = self.chasing_detector.validate_select(
            self.player, Piece(selected_piece_val), _selected, self.board
        )
        if not valid:
            chasing_mask = np.ones((self.height, self.width), dtype=bool)
            for _pos in position:
                if self.player == Player.BLUE:
                    _pos = (self.height - _pos[0] - 1, self.width - _pos[1] - 1)
                chasing_mask[_pos] = False

        if selected_piece_val == Piece.SCOUT.value:
            for direction in directions.T:
                positions = np.array(selected)[:, None] + direction[:, None]
                encountered_enemy = 0
                while (
                    np.all(positions >= 0, axis=0)
                    and np.all(positions < board_shape[:, None], axis=0)
                    and encountered_enemy < 1
                ):
                    if self.board[positions[0], positions[1]] != Piece.EMPTY.value:
                        if self.board[positions[0], positions[1]] > -Piece.FLAG.value:
                            break
                        encountered_enemy += 1
                    destinations[positions[0], positions[1]] = 1
                    positions += direction[:, None]

        else:
            positions = np.array(selected)[:, None] + directions
            valid_positions = positions[
                :,
                (np.all(positions >= 0, axis=0))
                & (np.all(positions < board_shape[:, None], axis=0)),
            ]
            mask = (
                self.board[valid_positions[0], valid_positions[1]] <= Piece.EMPTY.value
            ) & (
                self.board[valid_positions[0], valid_positions[1]] != -Piece.LAKE.value
            )
            valid_positions = valid_positions[:, mask]
            destinations[valid_positions[0], valid_positions[1]] = 1

        if two_square_mask is not None:
            destinations *= two_square_mask
        if chasing_mask is not None:
            destinations *= chasing_mask
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
