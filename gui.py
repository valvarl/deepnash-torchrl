import pygame
import gymnasium as gym
import time
import numpy as np
from deepnash.stratego_gym.envs.primitives import Piece, Player
from deepnash.stratego_gym.envs.startego import WINDOW_SIZE, GamePhase

from tests.conftest import env_5x5

def play_with_mouse():
    # env = gym.make("stratego_gym/Stratego-v0", render_mode="human")
    # env.reset()
    env = env_5x5()(pieces_num={Piece.FLAG: 1, Piece.SCOUT: 1}, render_mode="human")
    for i in range(0):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
    env.render()

    count = 0
    running = True
    start_time = time.time()
    while running and time.time() - start_time < 300:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos
                cell_size = (WINDOW_SIZE // env.board.shape[0], WINDOW_SIZE // env.board.shape[1])
                row = mouse_y // cell_size[1]
                col = mouse_x // cell_size[0]
                if env.game_phase == GamePhase.DEPLOY and count % 2 != 0 or env.game_phase != GamePhase.DEPLOY and (count // 2) % 2 != 0:
                    row = env.board.shape[0] - row - 1
                    col = env.board.shape[1] - col - 1
                print(row, col)
                count += 1

                state, reward, terminated, truncated, info = env.step((row, col))
                if terminated:
                    print(f"Game over! Player {-1 * info['cur_player']} received {reward}")
                    env.reset()

        env.render()
        time.sleep(0.1)
    
    pygame.quit()

if __name__ == "__main__":
    play_with_mouse()
