from enum import Enum

class Piece(Enum):
    SCOUT = "B"
    SOLDIER = "A"

class ChasingDetector:
    def __init__(self):
        self.chase_moves = []

    def is_adjacent(self, pos1, pos2):
        """Проверяет, являются ли клетки соседними по горизонтали или вертикали"""
        return (abs(pos1[0] - pos2[0]) == 1 and pos1[1] == pos2[1]) or \
               (abs(pos1[1] - pos2[1]) == 1 and pos1[0] == pos2[0])
    
    def is_on_same_line(self, pos1, pos2):
        """Проверяет, находятся ли две фигуры на одной линии"""
        return pos1[0] == pos2[0] or pos1[1] == pos2[1]

    def _step(self, piece_type, source, destination, is_chaser):
        if is_chaser:
            chased_pos = self.chase_moves[-1][1]
            chasing_pos = destination
            chasing_piece = piece_type
        else:
            chased_pos = source
            chasing_pos = self.chase_moves[-1][1]
            chasing_piece = self.chase_moves[-1][2]

        if chasing_piece != Piece.SCOUT.value:
            if self.is_adjacent(chased_pos, chasing_pos):
                self.chase_moves.append((source, destination, piece_type))
            else:
                self.chase_moves = [(source, destination, piece_type)]
        else:
            if self.is_on_same_line(chased_pos, chasing_pos):
                self.chase_moves.append((source, destination, piece_type))
            else:
                self.chase_moves = [(source, destination, piece_type)]
    
    def step(self, piece_type, source, destination):
        if len(self.chase_moves) == 0 or self.chase_moves[-1][1] == destination:
            self.chase_moves = [(source, destination, piece_type)]
            return
        
        is_chaser = False
        if len(self.chase_moves) > 1:
            if self.chase_moves[-2][1] != source:
                self.chase_moves = self.chase_moves[-1:]
            
            if (self.chase_moves[-1][2] != Piece.SCOUT.value and self.is_adjacent(self.chase_moves[-1][1], source)) or \
            (self.chase_moves[-1][2] == Piece.SCOUT.value and self.is_on_same_line(self.chase_moves[-1][1], source)):
                pass
            else:
                is_chaser = True

        self._step(piece_type, source, destination, is_chaser)
        print(self.chase_moves)

detector = ChasingDetector()

print(detector.step(Piece.SOLDIER.value, (0, 0), (0, 1)))  # Игрок A начинает преследование
print(detector.step(Piece.SOLDIER.value, (1, 1), (1, 2)))  # Игрок B ходит отдельно
print(detector.step(Piece.SOLDIER.value, (0, 1), (0, 2)))  # Игрок A продолжает преследование
print(detector.step(Piece.SOLDIER.value, (1, 2), (1, 1)))  # Игрок B делает неожиданный скачок
print(detector.step(Piece.SOLDIER.value, (0, 2), (0, 1)))  # Игрок A уходит с линии
print(detector.step(Piece.SOLDIER.value, (1, 1), (1, 0)))
print(detector.step(Piece.SOLDIER.value, (0, 1), (0, 0)))
print(detector.step(Piece.SOLDIER.value, (1, 0), (1, 1)))

print(detector.chase_moves)