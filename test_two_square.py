import numpy as np

from test_chasing import Player, Piece, Pos


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
        print(valid, positions)
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


detector = TwoSquareDetector()

detector.update(Player.RED, Piece.SCOUT, (0, 1), (0, 5))    
print(detector.p1)
detector.update(Player.RED, Piece.SCOUT, (0, 5), (0, 1))
print(detector.p1)
detector.update(Player.RED, Piece.SCOUT, (0, 1), (0, 3))
print(detector.p1)
detector.update(Player.RED, Piece.SCOUT, (0, 3), (0, -1))
print(detector.p1)
detector.update(Player.RED, Piece.SCOUT, (0, -1), (0, 3))
print(detector.p1)
detector.update(Player.RED, Piece.SCOUT, (0, 3), (0, -1))
print(detector.p1)
detector.update(Player.RED, Piece.SCOUT, (0, -1), (0, 3))
print(detector.p1)

# detector.update(Player.RED, Piece.SOLDIER, (0, 0), (0, 1))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (0, 1), (0, 0))
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (0, 0), (0, 1))
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (0, 1), (0, 0))
# print(detector.p1)

# detector.update(Player.RED, Piece.SOLDIER, (0, 0), (0, 1))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (0, 1), (0, 2))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (0, 2), (0, 3))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (0, 3), (0, 2))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (0, 2), (0, 1))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (0, 1), (0, 0))    
# print(detector.p1)

# detector.update(Player.RED, Piece.SOLDIER, (0, 0), (1, 0))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (1, 0), (2, 0))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (2, 0), (3, 0))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (3, 0), (2, 0))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (2, 0), (1, 0))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (1, 0), (0, 0))    
# print(detector.p1)

# detector.update(Player.RED, Piece.SOLDIER, (0, 0), (-1, 0))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (-1, 0), (-1, -1))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (-1, -1), (0, -1))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (-1, 0), (0, 0))    
# print(detector.p1)

# detector.update(Player.RED, Piece.SOLDIER, (0, 0), (-1, 0))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (-1, 0), (-2, 0))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (-2, 0), (-1, 0)) 
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (-1, 0), (0, 0))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (0, 0), (-1, 0))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (-1, 0), (0, 0))    
# print(detector.p1)
# detector.update(Player.RED, Piece.SOLDIER, (0, 0), (-1, 0))    
# print(detector.p1)