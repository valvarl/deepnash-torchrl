
from enum import Enum, auto


class Piece(Enum):
    EMPTY = auto()
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
    def __len__(cls):
        return cls.MARSHAL.value - 1

type Pos = tuple[int, int]