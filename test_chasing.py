import numpy as np

from stratego_gym.envs.detectors import ChasingDetector
from stratego_gym.envs.primitives import Player, Piece

detector = ChasingDetector()

# .B.
# R..
detector.update(Player.RED, Piece.SPY, (0, 0), (0, 1))
detector.update(Player.BLUE, Piece.SPY, (1, 1), (1, 2))
detector.update(Player.RED, Piece.SPY, (0, 1), (0, 2))
detector.update(Player.BLUE, Piece.SPY, (1, 2), (1, 1))
print(detector.validate_select(Player.RED, Piece.SPY, (0, 2)))
detector.update(Player.RED, Piece.SPY, (0, 2), (0, 1))
detector.update(Player.BLUE, Piece.SPY, (1, 1), (1, 0))
detector.update(Player.RED, Piece.SPY, (0, 1), (0, 0))
detector.update(Player.BLUE, Piece.SPY, (1, 0), (1, 1))
print(detector.validate_select(Player.RED, Piece.SPY, (0, 0)))
print(detector.validate_move(Player.RED, Piece.SPY, (0, 0), (0, 1)))
detector.update(Player.RED, Piece.SPY, (0, 0), (0, 1))

# ..B
# .R.
# detector.update(Player.RED, Piece.SPY, (0, 1), (0, 2))
# detector.update(Player.BLUE, Piece.SPY, (1, 2), (1, 1))
# detector.update(Player.RED, Piece.SPY, (0, 2), (0, 1))
# detector.update(Player.BLUE, Piece.SPY, (1, 1), (1, 0))
# detector.update(Player.RED, Piece.SPY, (0, 1), (0, 0))
# detector.update(Player.BLUE, Piece.SPY, (1, 0), (1, 1))
# detector.update(Player.RED, Piece.SPY, (0, 0), (0, 1))
# detector.update(Player.BLUE, Piece.SPY, (1, 1), (1, 2))
# print(detector.validate_select(Player.RED, Piece.SPY, (0, 1)))
# print(detector.validate_move(Player.RED, Piece.SPY, (0, 1), (0, 2)))
# detector.update(Player.RED, Piece.SPY, (0, 1), (0, 2))

# ..B
# .R.
# detector.update(Player.RED, Piece.SPY, (0, 1), (0, 2))
# detector.update(Player.BLUE, Piece.SPY, (1, 2), (1, 1))
# detector.update(Player.RED, Piece.SPY, (0, 2), (0, 1))
# detector.update(Player.BLUE, Piece.SPY, (1, 1), (1, 2))
# detector.update(Player.RED, Piece.SPY, (0, 1), (0, 2))
# detector.update(Player.BLUE, Piece.SPY, (1, 2), (1, 1))
# detector.update(Player.RED, Piece.SPY, (0, 2), (0, 1))
# detector.update(Player.BLUE, Piece.SPY, (1, 1), (1, 0))
# detector.update(Player.RED, Piece.SPY, (0, 1), (0, 0))
# detector.update(Player.BLUE, Piece.SPY, (1, 0), (1, 1))
# detector.update(Player.RED, Piece.SPY, (0, 0), (0, 1))


# .B.
# ...
# .R.
# detector.update(Player.RED, Piece.SPY, (-1, 1), (0, 1))
# detector.update(Player.BLUE, Piece.SPY, (1, 1), (1, 2))
# detector.update(Player.RED, Piece.SPY, (0, 1), (0, 2))
# detector.update(Player.BLUE, Piece.SPY, (1, 2), (1, 1))
# print(detector.validate_select(Player.RED, Piece.SPY, (0, 2)))
# detector.update(Player.RED, Piece.SPY, (0, 2), (0, 1))
# detector.update(Player.BLUE, Piece.SPY, (1, 1), (1, 0))
# detector.update(Player.RED, Piece.SPY, (0, 1), (0, 0))
# detector.update(Player.BLUE, Piece.SPY, (1, 0), (1, 1))
# print(detector.validate_select(Player.RED, Piece.SPY, (0, 0)))
# print(detector.validate_move(Player.RED, Piece.SPY, (0, 0), (0, 1)))
# detector.update(Player.RED, Piece.SPY, (0, 0), (0, 1))

# L.B
# .R.
# ..L
# detector.update(Player.RED, Piece.SPY, (0, 0), (0, 1))
# detector.update(Player.BLUE, Piece.SPY, (1, 1), (1, 0))
# detector.update(Player.RED, Piece.SPY, (0, 1), (1, 1))
# detector.update(Player.BLUE, Piece.SPY, (1, 0), (0, 0))
# detector.update(Player.RED, Piece.SPY, (1, 1), (1, 0))
# detector.update(Player.BLUE, Piece.SPY, (0, 0), (-1, 0))
# detector.update(Player.RED, Piece.SPY, (1, 0), (0, 0))
# detector.update(Player.BLUE, Piece.SPY, (-1, 0), (-1, -1))
# detector.update(Player.RED, Piece.SPY, (0, 0), (-1, 0))
# detector.update(Player.BLUE, Piece.SPY, (-1, -1), (0, -1))
# detector.update(Player.RED, Piece.SPY, (-1, 0), (-1, -1))
# detector.update(Player.BLUE, Piece.SPY, (0, -1), (0, 0))
# detector.update(Player.RED, Piece.SPY, (-1, -1), (0, -1))
# detector.update(Player.BLUE, Piece.SPY, (0, 0), (0, 1))
# detector.update(Player.RED, Piece.SPY, (0, -1), (0, 0))
# detector.update(Player.BLUE, Piece.SPY, (0, 1), (1, 1))
# print(detector.validate_move(Player.RED, Piece.SPY, (0, 0), (0, 1)))
# detector.update(Player.RED, Piece.SPY, (0, 0), (0, 1))

# ...
# ...
# R.B
# detector.update(Player.RED, Piece.SPY, (0, 0), (0, 1))
# detector.update(Player.BLUE, Piece.SPY, (0, 2), (1, 2))
# detector.update(Player.RED, Piece.SPY, (0, 1), (0, 2))
# detector.update(Player.BLUE, Piece.SPY, (1, 2), (2, 2))
# detector.update(Player.RED, Piece.SPY, (0, 2), (1, 2))
# detector.update(Player.BLUE, Piece.SPY, (2, 2), (2, 1))
# detector.update(Player.RED, Piece.SPY, (1, 2), (2, 2))
# detector.update(Player.BLUE, Piece.SPY, (2, 1), (2, 0))
# detector.update(Player.RED, Piece.SPY, (2, 2), (2, 1))
# detector.update(Player.BLUE, Piece.SPY, (2, 0), (1, 0))
# detector.update(Player.RED, Piece.SPY, (2, 1), (2, 0))
# detector.update(Player.BLUE, Piece.SPY, (1, 0), (0, 0))
# detector.update(Player.RED, Piece.SPY, (2, 0), (1, 0))
# detector.update(Player.BLUE, Piece.SPY, (0, 0), (0, 1))
# detector.update(Player.RED, Piece.SPY, (1, 0), (0, 0))
# detector.update(Player.BLUE, Piece.SPY, (0, 1), (0, 2))
# print(detector.chase_moves)
# print(detector.validate_select(Player.RED, Piece.SPY, (0, 0)))
# print(detector.validate_move(Player.RED, Piece.SPY, (0, 0), (0, 1)))
# detector.update(Player.RED, Piece.SPY, (0, 0), (0, 1))
# print(detector.chase_moves)

# R..
# ...
# ..B
# board = np.zeros((3, 3))
# board[1, 0] = 1
# detector.update(Player.RED, Piece.SCOUT, (2, 0), (2, 2), board)
# detector.update(Player.BLUE, Piece.SPY, (0, 2), (0, 1), board)
# detector.update(Player.RED, Piece.SCOUT, (2, 2), (2, 1), board)
# detector.update(Player.BLUE, Piece.SPY, (0, 1), (0, 0), board)
# print(detector.chase_moves)
# detector.update(Player.RED, Piece.SCOUT, (2, 1), (2, 0), board)
# print(detector.chase_moves)
# detector.update(Player.BLUE, Piece.SPY, (0, 0), (0, 1), board)
# detector.update(Player.RED, Piece.SCOUT, (2, 0), (2, 1), board)
# detector.update(Player.BLUE, Piece.SPY, (0, 1), (0, 2), board)
# detector.update(Player.RED, Piece.SCOUT, (2, 1), (2, 2), board)
# print(detector.chase_moves)
