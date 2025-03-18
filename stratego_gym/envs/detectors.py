from dataclasses import dataclass
from stratego_gym.envs.primitives import Piece, Player, Pos

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
    
    def validate_select(self, player: Player, piece: Piece, pos: Pos, board=None) -> tuple[bool, list[Pos] | None]:
        if not self.chase_moves or self.chase_moves[-1].attacker:
            return True, None
        forbidden = []
        for i, (chasing_move, chased_move) in enumerate(zip(self.chase_moves[::2], self.chase_moves[1::2])):
            if chasing_move.player == player and chasing_move.piece == piece and \
               chased_move.from_pos == self.chase_moves[-1].to_pos and \
               self.check_chasing_condition(piece, pos, chasing_move.to_pos, board):
                if (len(self.chase_moves) - 1) // 2 == i + 1 and self.chase_moves[-2].from_pos == chasing_move.to_pos:
                    continue
                else:
                    forbidden.append(chasing_move.to_pos)
        return (True, None) if not forbidden else (False, forbidden)

    def validate_move(self, player: Player, piece: Piece, from_pos: Pos, to_pos: Pos, board=None) -> bool:
        valid, forbidden = self.validate_select(player, piece, from_pos, board)
        if valid:
            return True
        if to_pos in forbidden:
            return False
        return True
    
    def update(self, player: Player, piece: Piece, from_pos: Pos, to_pos: Pos, board=None):
        if not self.chase_moves:
            self.chase_moves.append(ChaseEntry(player, piece, from_pos, to_pos))
            return
        
        if len(self.chase_moves) > 1 and self.chase_moves[-2].to_pos != from_pos:
            # selection of a figure not involved in chasing
            self.chase_moves = self.chase_moves[-1:]
            self.chase_moves[-1:][0].attacker = True
        elif self.chase_moves[-1].attacker and self.check_chasing_condition(piece, to_pos, self.chase_moves[-1].to_pos, board):
            # initiative handover
            self.chase_moves = [ChaseEntry(player, piece, from_pos, to_pos)]
            return

        if self.check_chasing_condition(
            verified_piece=self.chase_moves[-1].piece if self.chase_moves[-1].attacker else piece, 
            verified_pos=from_pos if self.chase_moves[-1].attacker else to_pos,
            opponent_pos=self.chase_moves[-1].to_pos,
            board=board
        ):
            # chase continues
            print(from_pos if self.chase_moves[-1].attacker else to_pos, self.chase_moves[-1])
            if not self.chase_moves[-1].attacker and not self.validate_move(player, piece, from_pos, to_pos, board):
                raise RuntimeError("")
            self.chase_moves.append(ChaseEntry(player, piece, from_pos, to_pos, attacker=not self.chase_moves[-1].attacker))
        else:
            self.chase_moves = [ChaseEntry(player, piece, from_pos, to_pos)]


class TwoSquareDetector:
    def __init__(self):
        self.p1: list[tuple[Pos, Pos]] = []
        self.p2: list[tuple[Pos, Pos]] = []

    def get_player(self, player: Player):
        return self.p1 if player == Player.RED else self.p2
    
    def validate_select(self, player: Player, piece: Piece, pos: Pos) -> tuple[bool, tuple[Pos, Pos] | None]:
        p = self.get_player(player)
        if len(p) < 3:
            return True, None
        if pos != p[-1][1]:
            return True, None
        if piece != Piece.SCOUT:
            return False, (p[-1][0], p[-1][0])
        start_pos, end_pos = p[0]
        if start_pos[0] == end_pos[0]:
            if start_pos[1] < end_pos[1]:
                start_pos = (start_pos[0], min(start_pos[1], p[1][1][1]))
                end_pos = (end_pos[0], max(end_pos[1], p[2][1][1]))
                return False, (start_pos, pos)
            else:
                start_pos = (start_pos[0], max(start_pos[1], p[1][1][1]))
                end_pos = (end_pos[0], min(end_pos[1], p[2][1][1]))
                return False, (pos, start_pos)
        else:
            if start_pos[1] < end_pos[1]:
                start_pos = (min(start_pos[0], p[1][1][0]), start_pos[1])
                end_pos = (max(end_pos[0], p[2][1][0]), end_pos[1])
                return False, (start_pos, pos)
            else:
                start_pos = (max(start_pos[0], p[1][1][0]), start_pos[1])
                end_pos = (min(end_pos[0], p[2][1][0]), end_pos[1])
                return False, (pos, start_pos)

    def validate_move(self, player: Player, piece: Piece, from_pos: Pos, to_pos: Pos) -> bool:
        valid, positions = self.validate_select(player, piece, from_pos)
        if valid:
            return True
        start_pos, end_pos = positions
        idx = 1 if start_pos[0] == end_pos[0] else 0
        if to_pos[1-idx] == start_pos[1-idx]:
            if start_pos[idx] <= to_pos[idx] <= end_pos[idx] or end_pos[idx] <= to_pos[idx] <= start_pos[idx]:
                return False
        return True

    def update(self, player: Player, piece: Piece, from_pos: Pos, to_pos: Pos):
        p = self.get_player(player)
        if not p:
            p.append((from_pos, to_pos))
            return
        
        if from_pos != p[-1][1]:
            p.clear()
            p.append((from_pos, to_pos))
            return
        
        start_pos, end_pos = p[-1]
        idx = 1 if start_pos[0] == end_pos[0] else 0
        if start_pos[idx] < end_pos[idx] and start_pos[idx] <= to_pos[idx] < end_pos[idx] or \
           start_pos[idx] > end_pos[idx] and end_pos[idx] < to_pos[idx] <= start_pos[idx]:
            if self.validate_move(player, piece, from_pos, to_pos):
                p.append((from_pos, to_pos))
            else:
                raise RuntimeError("")
        else:
            p.clear()
            p.append((from_pos, to_pos))

        # print('P1', self.p1)
        # print('P2', self.p2)
