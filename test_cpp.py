import sys
import os

from stratego.wrappers.cpp_env import StrategoEnvCpp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "python")))

import numpy as np
from stratego.wrappers.cpp_config import StrategoConfigCpp
from stratego.core.primitives import Piece

from stratego.cpp import stratego_cpp as sp

# Define pieces for each player (Spy and Flag only)
p1_pieces = {Piece.SPY: 1, Piece.FLAG: 1}

p2_pieces = {Piece.SPY: 1, Piece.FLAG: 1}

init_from_mask = False

if not init_from_mask:
    # Define lake position (center 2x2 area)
    lakes = [((1, 1), (2, 2))]

    # Define deployment areas (top and bottom rows)
    p1_deploy = [((3, 0), (3, 3))]

    p2_deploy = [((0, 0), (0, 3))]
    # Create config

    config = StrategoConfigCpp(
        height=4,
        width=4,
        p1_pieces=p1_pieces,
        p2_pieces=p2_pieces,
        lakes=lakes,
        p1_places_to_deploy=p1_deploy,
        p2_places_to_deploy=p2_deploy,
        total_moves_limit=100,
        moves_since_attack_limit=20,
        observed_history_entries=10,
    )
else:
    height = 4
    width = 4

    lakes_mask = np.zeros((4, 4), dtype=bool)
    lakes_mask[1:3, 1:3] = True
    p1_deploy_mask = np.zeros((4, 4), dtype=bool)
    p1_deploy_mask[3] = True
    p2_deploy_mask = np.zeros((4, 4), dtype=bool)
    p2_deploy_mask[0] = True

    config = StrategoConfigCpp(
        height=4,
        width=4,
        p1_pieces=p1_pieces,
        p2_pieces=p2_pieces,
        lakes_mask=lakes_mask,
        p1_deploy_mask=p1_deploy_mask,
        p2_deploy_mask=p2_deploy_mask,
        total_moves_limit=100,
        moves_since_attack_limit=20,
        observed_history_entries=10,
    )

env = StrategoEnvCpp(config)
print(env.game_phase)
env.reset()
print(env.game_phase)
obs, action_mask, reward, term, trunc = env.step([3, 0])
print(action_mask)
print(env.board, env.current_player)
obs, action_mask, reward, term, trunc = env.step([3, 0])
print(action_mask)
print(env.board, env.current_player)
env.step([3, 1])
print(env.board, env.current_player)
obs, action_mask, reward, term, trunc = env.step([3, 1])
print(action_mask)
print(env.board, env.current_player)
print(env.game_phase)
obs, action_mask, reward, term, trunc = env.step([3, 1])
print(action_mask)
print(env.game_phase)
obs, action_mask, reward, term, trunc = env.step([3, 2])
print(action_mask)
print(env.board, env.current_player)
print(env.game_phase)

env.step([3, 1])
obs, action_mask, reward, term, trunc = env.step([3, 2])
print(action_mask)
print(env.board, env.current_player)
print(env.game_phase)

env.step([3, 2])
env.step([3, 3])
env.step([3, 2])
env.step([3, 3])

env.step([3, 3])
env.step([2, 3])
env.step([3, 3])
env.step([2, 3])

env.step([2, 3])
env.step([1, 3])
env.step([2, 3])
env.step([1, 3])

env.step([1, 3])
obs, action_mask, reward, term, trunc = env.step([0, 3])
print(action_mask)
print(reward, term, trunc)
print(env.board, env.current_player)
print(env.game_phase)
print(env.get_info())
