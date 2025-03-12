
from enum import Enum, auto


class Piece(Enum):
    EMPTY = 0
    LAKE = 1
    FLAG = 2
    BOMB = 3
    SPY = 4
    SCOUT = 5
    MINER = 6
    SERGEANT = 7
    LIEUTENANT = 8
    CAPTAIN = 9
    MAJOR = 10
    COLONEL = 11
    GENERAL = 12
    MARSHAL = 13

    @classmethod
    def unique_pieces_num(cls):
        return cls.MARSHAL.value - cls.FLAG.value + 1

type Pos = tuple[int, int]
