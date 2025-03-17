from dataclasses import dataclass
from enum import Enum

import numpy as np

class Player:
    RED = 0
    BLUE = 1

class Piece(Enum):
    SCOUT = "B"
    SOLDIER = "A"

type Pos = tuple[int, int]

@dataclass
class ChaseEntry:
    player: Player
    piece: Piece
    from_pos: Pos
    to_pos: Pos
    attacker: bool = True


class ChasingDetector:
    def __init__(self):
        self.chase_moves: list[ChaseEntry] = []

    def is_adjacent(self, pos1, pos2) -> bool:
        return (abs(pos1[0] - pos2[0]) == 1 and pos1[1] == pos2[1]) or \
               (abs(pos1[1] - pos2[1]) == 1 and pos1[0] == pos2[0])
    
    def is_on_same_line(self, pos1, pos2, board=None) -> bool:
        if pos1[0] == pos2[0]:
            return (board[pos1[0]][min(pos1[1], pos2[1]) + 1: max(pos1[1], pos2[1])] == 0).all()
        if pos1[1] == pos2[1]:
            return (board[:, pos1[1]][min(pos1[0], pos2[0]) + 1: max(pos1[0], pos2[0])] == 0).all()
        return False
    
    def check_chasing_condition(self, verified_piece: Piece,  verified_pos: Pos, opponent_pos: Pos, board=None) -> bool:
        if verified_piece != Piece.SCOUT:
            return self.is_adjacent(verified_pos, opponent_pos)
        else:
            return self.is_on_same_line(verified_pos, opponent_pos, board)
    
    def validate_select(self, player: Player, piece: Piece, pos: Pos, board=None) -> tuple[bool, Pos | None]:
        if not self.chase_moves or self.chase_moves[-1].attacker:
            return True, None
        for i, (chasing_move, chased_move) in enumerate(zip(self.chase_moves[::2], self.chase_moves[1::2])):
            if chasing_move.player == player and chasing_move.piece == piece and \
               chased_move.from_pos == self.chase_moves[-1].to_pos and \
               self.check_chasing_condition(piece, pos, chasing_move.to_pos, board) and \
               self.check_chasing_condition(piece, chasing_move.from_pos, chasing_move.to_pos, board):
                if len(self.chase_moves) > 1 and (len(self.chase_moves) - 1) // 2 == i + 1 and self.chase_moves[-2].from_pos == chasing_move.to_pos:
                    return True, None
                else:
                    return False, chasing_move.to_pos
        return True, None

    def validate_move(self, player: Player, piece: Piece, from_pos: Pos, to_pos: Pos, board=None) -> bool:
        valid, position = self.validate_select(player, piece, from_pos, board)
        if valid:
            return True
        if position == to_pos:
            return False
        return True
    
    def update(self, player: Player, piece: Piece, from_pos: Pos, to_pos: Pos, board=None):
        if not self.chase_moves:
            self.chase_moves.append(ChaseEntry(player, piece, from_pos, to_pos))
            return
        
        if len(self.chase_moves) > 1 and self.chase_moves[-2].to_pos != from_pos:
            self.chase_moves = self.chase_moves[-1:]
            self.chase_moves[-1:][0].attacker = True

        if self.check_chasing_condition(
            verified_piece=self.chase_moves[-1].piece if self.chase_moves[-1].attacker else piece, 
            verified_pos=from_pos if self.chase_moves[-1].attacker else to_pos,
            opponent_pos=self.chase_moves[-1].to_pos,
            board=board
        ):
            if not self.chase_moves[-1].attacker and not self.validate_move(player, piece, from_pos, to_pos, board):
                raise RuntimeError("")
            self.chase_moves.append(ChaseEntry(player, piece, from_pos, to_pos, attacker=not self.chase_moves[-1].attacker))
        else:
            self.chase_moves = [ChaseEntry(player, piece, from_pos, to_pos)]

    def clear(self):
        self.chase_moves = []

detector = ChasingDetector()

# .B.
# R..
# detector.update(Player.RED, Piece.SOLDIER, (0, 0), (0, 1))
# detector.update(Player.BLUE, Piece.SOLDIER, (1, 1), (1, 2))
# detector.update(Player.RED, Piece.SOLDIER, (0, 1), (0, 2))
# detector.update(Player.BLUE, Piece.SOLDIER, (1, 2), (1, 1))
# print(detector.validate_select(Player.RED, Piece.SOLDIER, (0, 2)))
# detector.update(Player.RED, Piece.SOLDIER, (0, 2), (0, 1))
# detector.update(Player.BLUE, Piece.SOLDIER, (1, 1), (1, 0))
# detector.update(Player.RED, Piece.SOLDIER, (0, 1), (0, 0))
# detector.update(Player.BLUE, Piece.SOLDIER, (1, 0), (1, 1))
# print(detector.validate_select(Player.RED, Piece.SOLDIER, (0, 0)))
# print(detector.validate_move(Player.RED, Piece.SOLDIER, (0, 0), (0, 1)))
# detector.update(Player.RED, Piece.SOLDIER, (0, 0), (0, 1))

# ..B
# .R.
# detector.update(Player.RED, Piece.SOLDIER, (0, 1), (0, 2))
# detector.update(Player.BLUE, Piece.SOLDIER, (1, 2), (1, 1))
# detector.update(Player.RED, Piece.SOLDIER, (0, 2), (0, 1))
# detector.update(Player.BLUE, Piece.SOLDIER, (1, 1), (1, 0))
# detector.update(Player.RED, Piece.SOLDIER, (0, 1), (0, 0))
# detector.update(Player.BLUE, Piece.SOLDIER, (1, 0), (1, 1))
# detector.update(Player.RED, Piece.SOLDIER, (0, 0), (0, 1))
# detector.update(Player.BLUE, Piece.SOLDIER, (1, 1), (1, 2))
# print(detector.validate_select(Player.RED, Piece.SOLDIER, (0, 1)))
# print(detector.validate_move(Player.RED, Piece.SOLDIER, (0, 1), (0, 2)))
# detector.update(Player.RED, Piece.SOLDIER, (0, 1), (0, 2))

# ..B
# .R.
# detector.update(Player.RED, Piece.SOLDIER, (0, 1), (0, 2))
# detector.update(Player.BLUE, Piece.SOLDIER, (1, 2), (1, 1))
# detector.update(Player.RED, Piece.SOLDIER, (0, 2), (0, 1))
# detector.update(Player.BLUE, Piece.SOLDIER, (1, 1), (1, 2))
# detector.update(Player.RED, Piece.SOLDIER, (0, 1), (0, 2))
# detector.update(Player.BLUE, Piece.SOLDIER, (1, 2), (1, 1))
# detector.update(Player.RED, Piece.SOLDIER, (0, 2), (0, 1))

# .B.
# ...
# .R.
detector.update(Player.RED, Piece.SOLDIER, (-1, 1), (0, 1))
detector.update(Player.BLUE, Piece.SOLDIER, (1, 1), (1, 2))
detector.update(Player.RED, Piece.SOLDIER, (0, 1), (0, 2))
detector.update(Player.BLUE, Piece.SOLDIER, (1, 2), (1, 1))
print(detector.validate_select(Player.RED, Piece.SOLDIER, (0, 2)))
detector.update(Player.RED, Piece.SOLDIER, (0, 2), (0, 1))
detector.update(Player.BLUE, Piece.SOLDIER, (1, 1), (1, 0))
detector.update(Player.RED, Piece.SOLDIER, (0, 1), (0, 0))
detector.update(Player.BLUE, Piece.SOLDIER, (1, 0), (1, 1))
print(detector.validate_select(Player.RED, Piece.SOLDIER, (0, 0)))
print(detector.validate_move(Player.RED, Piece.SOLDIER, (0, 0), (0, 1)))
detector.update(Player.RED, Piece.SOLDIER, (0, 0), (0, 1))

# L.B
# .R.
# ..L
# detector.update(Player.RED, Piece.SOLDIER, (0, 0), (0, 1))
# detector.update(Player.BLUE, Piece.SOLDIER, (1, 1), (1, 0))
# detector.update(Player.RED, Piece.SOLDIER, (0, 1), (1, 1))
# detector.update(Player.BLUE, Piece.SOLDIER, (1, 0), (0, 0))
# detector.update(Player.RED, Piece.SOLDIER, (1, 1), (1, 0))
# detector.update(Player.BLUE, Piece.SOLDIER, (0, 0), (-1, 0))
# detector.update(Player.RED, Piece.SOLDIER, (1, 0), (0, 0))
# detector.update(Player.BLUE, Piece.SOLDIER, (-1, 0), (-1, -1))
# detector.update(Player.RED, Piece.SOLDIER, (0, 0), (-1, 0))
# detector.update(Player.BLUE, Piece.SOLDIER, (-1, -1), (0, -1))
# detector.update(Player.RED, Piece.SOLDIER, (-1, 0), (-1, -1))
# detector.update(Player.BLUE, Piece.SOLDIER, (0, -1), (0, 0))
# detector.update(Player.RED, Piece.SOLDIER, (-1, -1), (0, -1))
# detector.update(Player.BLUE, Piece.SOLDIER, (0, 0), (0, 1))
# detector.update(Player.RED, Piece.SOLDIER, (0, -1), (0, 0))
# detector.update(Player.BLUE, Piece.SOLDIER, (0, 1), (1, 1))
# print(detector.validate_move(Player.RED, Piece.SOLDIER, (0, 0), (0, 1)))
# detector.update(Player.RED, Piece.SOLDIER, (0, 0), (0, 1))


# ..B
# .L.
# R..
# board = np.zeros((3, 3))
# board[1, 1] = 1
# detector.update(Player.RED, Piece.SCOUT, (0, 0), (0, 2), board)
# detector.update(Player.BLUE, Piece.SCOUT, (2, 2), (2, 0), board)
# detector.update(Player.RED, Piece.SCOUT, (0, 2), (2, 2), board)
# detector.update(Player.BLUE, Piece.SCOUT, (2, 0), (0, 0), board)
# detector.update(Player.RED, Piece.SCOUT, (2, 2), (2, 0), board)
# detector.update(Player.BLUE, Piece.SCOUT, (0, 0), (0, 2), board)
# detector.update(Player.RED, Piece.SCOUT, (2, 0), (0, 0), board)
# detector.update(Player.BLUE, Piece.SCOUT, (0, 2), (2, 2), board)
# print(detector.chase_moves)
# print(detector.validate_select(Player.RED, Piece.SCOUT, (0, 0), board))
# print(detector.validate_move(Player.RED, Piece.SCOUT, (0, 0), (0, 2), board))
# detector.update(Player.RED, Piece.SCOUT, (0, 0), (0, 2), board)
# print(detector.chase_moves)


# ...
# ...
# R.B
# detector.update(Player.RED, Piece.SOLDIER, (0, 0), (0, 1))
# detector.update(Player.BLUE, Piece.SOLDIER, (0, 2), (1, 2))
# detector.update(Player.RED, Piece.SOLDIER, (0, 1), (0, 2))
# detector.update(Player.BLUE, Piece.SOLDIER, (1, 2), (2, 2))
# detector.update(Player.RED, Piece.SOLDIER, (0, 2), (1, 2))
# detector.update(Player.BLUE, Piece.SOLDIER, (2, 2), (2, 1))
# detector.update(Player.RED, Piece.SOLDIER, (1, 2), (2, 2))
# detector.update(Player.BLUE, Piece.SOLDIER, (2, 1), (2, 0))
# detector.update(Player.RED, Piece.SOLDIER, (2, 2), (2, 1))
# detector.update(Player.BLUE, Piece.SOLDIER, (2, 0), (1, 0))
# detector.update(Player.RED, Piece.SOLDIER, (2, 1), (2, 0))
# detector.update(Player.BLUE, Piece.SOLDIER, (1, 0), (0, 0))
# detector.update(Player.RED, Piece.SOLDIER, (2, 0), (1, 0))
# detector.update(Player.BLUE, Piece.SOLDIER, (0, 0), (0, 1))
# detector.update(Player.RED, Piece.SOLDIER, (1, 0), (0, 0))
# detector.update(Player.BLUE, Piece.SOLDIER, (0, 1), (0, 2))
# print(detector.chase_moves)
# print(detector.validate_select(Player.RED, Piece.SOLDIER, (0, 0)))
# print(detector.validate_move(Player.RED, Piece.SOLDIER, (0, 0), (0, 1)))
# detector.update(Player.RED, Piece.SOLDIER, (0, 0), (0, 1))
# print(detector.chase_moves)

# R..
# ...
# ..B
# board = np.zeros((3, 3))
# board[1, 0] = 1
# detector.update(Player.RED, Piece.SCOUT, (2, 0), (2, 2), board)
# detector.update(Player.BLUE, Piece.SOLDIER, (0, 2), (0, 1), board)
# detector.update(Player.RED, Piece.SCOUT, (2, 2), (2, 1), board)
# detector.update(Player.BLUE, Piece.SOLDIER, (0, 1), (0, 0), board)
# print(detector.chase_moves)
# detector.update(Player.RED, Piece.SCOUT, (2, 1), (2, 0), board)
# print(detector.chase_moves)
# detector.update(Player.BLUE, Piece.SOLDIER, (0, 0), (0, 1), board)
# detector.update(Player.RED, Piece.SCOUT, (2, 0), (2, 1), board)
# detector.update(Player.BLUE, Piece.SOLDIER, (0, 1), (0, 2), board)
# detector.update(Player.RED, Piece.SCOUT, (2, 1), (2, 2), board)
# print(detector.chase_moves)