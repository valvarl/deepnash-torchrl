import torch
import torchrl.envs.libs.gym
from tensordict import TensorDict
from torchrl.collectors import (
    MultiSyncDataCollector,
    MultiaSyncDataCollector,
    SyncDataCollector,
)
from torchrl.data import LazyTensorStorage, SliceSampler, ReplayBuffer
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import default_info_dict_reader, ParallelEnv

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np
import time

from deepnash import DeepNashAgent
from stratego.core.primitives import Piece, Player


def basic_test():
    import stratego

    env = gym.make("stratego_gym/StrategoCpp-v0", render_mode="human")
    env.reset()

    start_time = time.time()
    count = 0
    games = 0
    while time.time() - start_time < 60:
        # try:
        #     action = env.action_space.sample()
        # except Exception:
        #     print(env.board)
        #     print(env.chasing_detector.chase_moves)
        #     print("p1", env.two_square_detector.p1)
        #     print("p2", env.two_square_detector.p2)
        #     print(env.player)
        #     print(env.p2.last_selected)
        #     print(env.p2.last_selected_piece)
        #     pos = env.two_square_detector.p2[-1][1]
        #     piece = Piece(env.board[pos])
        #     print(env.two_square_detector.validate_select(Player.BLUE, piece, pos))
        #     print(env.valid_pieces_to_select())

        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        # try:
        #     state, reward, terminated, truncated, info = env.step(action)
        #     action_mask = state["action_mask"]
        # except ValueError as e:
        #     print(info["cur_board"])
        #     print(action_mask)
        #     raise e

        # env.render()
        # time.sleep(0.1)
        count += 1
        if terminated:
            games += 1
            print(
                f"Game over! Player {-1 * info['cur_player']} received {reward}, game: {games}, turn: {info['total_moves']}, turn: {info['moves_since_attack']}"
            )
            # if reward == 0 and not (
            #     info["total_moves"] == 2000 or info["moves_since_attack"] == 200
            # ):
            #     # print(env.draw_conditions)
            #     print(env.board)
            #     print("p1", env.two_square_detector.p1)
            #     print("p2", env.two_square_detector.p2)
            #     print(env.player)
            #     exit()
            env.reset()
    print(count)


def policy_test():
    device = "cuda"
    agent = DeepNashAgent().to(device)
    # agent = torch.load("DeepNashPolicy.pt").to(device)

    reader = default_info_dict_reader(["cur_player"])
    env = (
        GymEnv("stratego_gym/StrategoCpp-v0", render_mode=None)
        .set_info_dict_reader(reader)
        .to(device)
    )
    tensordict = env.reset()

    start_time = time.time()
    count = 0
    while time.time() - start_time < 60:
        tensordict = agent(tensordict)
        tensordict = env.step(tensordict)["next"]
        # env.render()
        # time.sleep(0.1)
        count += 1
        if tensordict["terminated"]:
            info = env.get_info()
            print(
                f"Game over! Player {-1 * tensordict['cur_player']} received {tensordict['reward']}, turn: {info['total_moves']}, turn: {info['moves_since_attack']}"
            )
            tensordict = env.reset()
    print(count)


def rollout_test():
    device = "cuda"
    agent = DeepNashAgent().to(device)

    reader = default_info_dict_reader(["cur_player"])
    env = (
        GymEnv("stratego_gym/StrategoCpp-v0", render_mode=None)
        .set_info_dict_reader(reader)
        .to(device)
    )

    start_time = time.time()
    count = 0
    while time.time() - start_time < 60:
        tensordict_rollout = env.rollout(max_steps=3600, policy=agent)
        count += tensordict_rollout.batch_size[0]
        tensordict_rollout = tensordict_rollout[-1]["next"]
        if tensordict_rollout["terminated"]:
            info = env.get_info()
            print(
                f"Game over! Player {-1 * tensordict_rollout['cur_player']} received {tensordict_rollout['reward']}, turn: {info['total_moves']}, turn: {info['moves_since_attack']}"
            )
    print(count)


def make_env():
    reader = default_info_dict_reader(["cur_player"])
    return GymEnv("stratego_gym/StrategoCpp-v0").set_info_dict_reader(reader)


def vectorized_test(n_procs, n_workers):
    device = "cuda"
    N = 2000
    policy = DeepNashAgent()
    env = make_env()
    policy(env.reset())

    memory = ReplayBuffer(
        storage=LazyTensorStorage(N, ndim=2),
        sampler=SliceSampler(num_slices=4, traj_key=("collector", "traj_ids")),
    )
    collector = MultiaSyncDataCollector(
        [ParallelEnv(n_workers, make_env)] * n_procs,
        policy,
        frames_per_batch=N,
        total_frames=-1,
        cat_results="stack",
        device=device,
    )

    start_time = time.time()
    count = 0
    for data in collector:
        print(data.shape)
        memory.extend(data)
        print("Batch Added to Replay Buffer: " + str(count))
        count += 1
        if (time.time() - start_time) > 60:
            break


if __name__ == "__main__":
    vectorized_test(4, 4)
    # policy_test()
    # basic_test()
    # rollout_test()
