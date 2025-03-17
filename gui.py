import pygame
import gymnasium as gym
import time
import numpy as np
from stratego_gym.envs.primitives import Piece, Player
from stratego_gym.envs.startego import WINDOW_SIZE, GamePhase

def play_with_mouse():
    env = gym.make("stratego_gym/Stratego-v0", render_mode="human")
    env.reset()

    for i in range(80):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.1)

    selected_piece = None
    running = True

    start_time = time.time()
    count = 0
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