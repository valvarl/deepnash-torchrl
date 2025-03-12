
from enum import Enum, auto


class Piece(Enum):
    EMPTY = 0
    LAKE = auto()
    FLAG = auto()
    BOMB = auto()
    SPY = auto()
    SCOUT = auto()
    MINER = auto()
    SERGEANT = auto()
    LIEUTENANT = auto()
    CAPTAIN = auto()
    MAJOR = auto()
    COLONEL = auto()
    GENERAL = auto()
    MARSHAL = auto()

    @classmethod
    def unique_pieces_num(cls):
        return cls.MARSHAL.value - cls.FLAG.value + 1

type Pos = tuple[int, int]
