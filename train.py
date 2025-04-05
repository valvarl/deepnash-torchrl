
import logging
import numpy as np
from tensordict.nn import TensorDictModule
import torch
from torchrl.envs import default_info_dict_reader
from torchrl.envs.libs.gym import GymEnv
from tqdm import tqdm

from deepnash.nn import DeepNashAgent
from deepnash.learn.rnad import RNaDSolver
from deepnash.learn.config import RNaDConfig
from deepnash.stratego_gym.envs.config import StrategoConfig
from deepnash.stratego_gym.envs.primitives import Piece
from deepnash.stratego_gym.envs.startego import GamePhase  # must be importable on the driver side too

MAP_4x4 = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]], dtype=bool)

LIMITED_PIECE_SET = {Piece.FLAG: 1, Piece.SPY: 1, Piece.SCOUT: 1, Piece.MARSHAL: 1}
PLACES_TO_DEPLOY_RED = [((3, 0), (3, 3)),]
PLACES_TO_DEPLOY_BLUE = [((0, 0), (0, 3)),]
TEST_CONFIG = StrategoConfig(
    height=4,
    width=4,
    p1_pieces=LIMITED_PIECE_SET,
    p1_places_to_deploy=PLACES_TO_DEPLOY_RED,
    p2_places_to_deploy=PLACES_TO_DEPLOY_BLUE,
    lakes_mask=MAP_4x4,
)

# 1. Create environment factory
def env_maker(config=TEST_CONFIG, render_mode=None):
    reader = default_info_dict_reader(["cur_player", "game_phase"])
    return GymEnv("stratego_gym/Stratego-v0", config=config, render_mode=render_mode).set_info_dict_reader(reader)

@torch.no_grad
def evaluate_random(policy: TensorDictModule):
    env = env_maker().to("cuda")
    device = policy.device
    policy = policy.to("cuda")
    env.set_info_dict_reader(default_info_dict_reader(["cur_player", "game_phase", "cur_board"]))
    win_count = 0
    draw_count = 0
    for i in tqdm(range(100)):
        tensordict = env.reset()
        policy_turn = np.random.choice([1, -1])
        # print(f"New Game! {i}")
        while True:
            # if i < 10: print("------------------------")
            # if i < 10: print(f"Game Phase: {tensordict['game_phase'].item()}")
            # if i < 10: print(f"Cur Player: {tensordict['cur_player'].item()}")
            game_phase = tensordict['game_phase'].item()

            try:
                if tensordict["cur_player"] == policy_turn:
                    tensordict = policy(tensordict)
                    # if i < 10: print(f"Value: {tensordict['value'].item()}")
                    # if i < 10: print(tensordict["policy"])
                    tensordict = env.step(tensordict)["next"]
                else:
                    # if i < 10: print("Random Action")
                    tensordict["action"] = env.action_spec.sample()
                    tensordict = env.step(tensordict)["next"]
            except RuntimeError as e:
                print(env.board)
                print(env.valid_pieces_to_select(debug_print=True))
                print(env.game_phase)
                print(env.player)
                print(env.two_square_detector.p1)
                print(env.two_square_detector.p2)
                print(env.chasing_detector.chase_moves)
                raise e

            # if game_phase in (GamePhase.DEPLOY, GamePhase.MOVE):
            #     if i < 10: print(tensordict["cur_board"])

            if tensordict["terminated"]:
                # if i < 10: print(f"Won Game? {-policy_turn == tensordict['cur_player']}")
                # if i < 10: print(f"Draw Game? {(tensordict['reward'] == 0).item()}")
                win_count += int(-policy_turn == tensordict["cur_player"]) * tensordict["reward"].item()
                draw_count += 1 - tensordict["reward"].item()
                break

    env.close()

    policy = policy.to(device)
    return {
        "win": win_count / 100,
        "draw": draw_count / 100,
        "loss": (100 - (win_count + draw_count)) / 100,
    }


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(level=logging.DEBUG)

    solver = RNaDSolver(config=RNaDConfig(game_name="stratego"), wandb=True, directory_name="250326_2357")
    solver.run(env_maker, evaluate_fn=evaluate_random)

    # for name, param in policy.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.data}")
    # breakpoint()
