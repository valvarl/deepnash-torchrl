import pygame
import gymnasium as gym
import time
import numpy as np
import torch
from torchrl.envs import default_info_dict_reader
from deepnash.nn.agent import DeepNashAgent
from deepnash.stratego_gym.envs.config import StrategoConfig
from deepnash.stratego_gym.envs.primitives import Piece, Player
from deepnash.stratego_gym.envs.startego import WINDOW_SIZE, GamePhase, StrategoEnv

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

def play_with_mouse():
    # env = gym.make("stratego_gym/Stratego-v0", render_mode="human")
    # env.reset()
    # env = env_5x5()(pieces_num={Piece.FLAG: 1, Piece.SCOUT: 1}, render_mode="human")
    from train import env_maker
    env = env_maker(render_mode="human")
    tensordict = env.reset()

    saved_dict = torch.load("saved_runs/250326_2357/22/60", weights_only=False)
    policy = DeepNashAgent(device="cpu")
    policy.eval()
    policy.load_state_dict(saved_dict["policy"])

    # for i in range(0):
    #     action = env.action_space.sample()
    #     state, reward, terminated, truncated, info = env.step(action)
    env.render()

    count = 0
    running = True
    start_time = time.time()
    while running and time.time() - start_time < 300:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif env.player == Player.BLUE:
                # print(tensordict["obs"])
                tensordict = policy(tensordict)
                tensordict = env.step(tensordict)["next"]

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos
                cell_size = (WINDOW_SIZE // env.board.shape[0], WINDOW_SIZE // env.board.shape[1])
                row = mouse_y // cell_size[1]
                col = mouse_x // cell_size[0]
                # if env.game_phase == GamePhase.DEPLOY and count % 2 != 0 or env.game_phase != GamePhase.DEPLOY and (count // 2) % 2 != 0:
                #     row = env.board.shape[0] - row - 1
                #     col = env.board.shape[1] - col - 1
                print(row, col)
                count += 1

                act = torch.zeros((2, 4))
                act[0, row] = 1
                act[1, col] = 1
                tensordict["action"] = act
                print(tensordict["action"])
                tensordict = env.step(tensordict)["next"]
            if tensordict["terminated"]:
                print(f"Game over! Player {-1 * tensordict['cur_player']} received {tensordict["reward"]}")
                tensordict = env.reset()

        env.render()
        time.sleep(0.1)
    
    pygame.quit()

if __name__ == "__main__":
    play_with_mouse()
