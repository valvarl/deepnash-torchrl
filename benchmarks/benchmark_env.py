import argparse
from functools import partial
import logging
import os
import tempfile
from time import perf_counter

import gymnasium as gym
import torch
from torchrl.collectors import (
    MultiSyncDataCollector,
    MultiaSyncDataCollector,
    SyncDataCollector,
)
from torchrl.data import LazyTensorStorage, SliceSampler, ReplayBuffer
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import default_info_dict_reader, ParallelEnv

from deepnash import DeepNashAgent


def setup_logger(log_to_file=False, filename="benchmark_log.txt"):
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if log_to_file:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def basic_test(env_name, duration, log_to_file=False):
    logger = setup_logger(log_to_file)
    env = gym.make(env_name, render_mode=None)
    env.reset()

    start_time = perf_counter()
    count = 0
    games = 0
    turns_per_game = []

    logger.info(f"Starting basic_test on '{env_name}' for {duration}s...")

    while perf_counter() - start_time < duration:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        count += 1
        if terminated:
            games += 1
            turns = int(info["total_moves"])
            turns_per_game.append(turns)
            logger.info(
                f"Game {games}: Player {-1 * info['cur_player']} received {reward}, "
                + f"total_moves: {turns}, moves_since_attack: {int(info['moves_since_attack'])}"
            )
            env.reset()

    elapsed = perf_counter() - start_time
    fps = count / elapsed
    avg_game_length = sum(turns_per_game) / len(turns_per_game) if turns_per_game else 0

    logger.info("=== Benchmark Summary ===")
    logger.info(f"Total steps: {count}")
    logger.info(f"Elapsed time: {elapsed:.2f}s")
    logger.info(f"Steps per second (FPS): {fps:.2f}")
    logger.info(f"Games played: {games}")
    logger.info(f"Average game length: {avg_game_length:.2f} turns")

    return {
        "steps": count,
        "games": games,
        "fps": fps,
        "avg_game_length": avg_game_length,
        "duration": elapsed,
    }


def make_env(env_name):
    reader = default_info_dict_reader(["cur_player"])
    return GymEnv(env_name, render_mode=None).set_info_dict_reader(reader)


def policy_test(env_name, device, compile_policy, duration, log_to_file=False):
    logger = setup_logger(log_to_file)
    logger.info(
        f"Starting policy_test on '{env_name}' for {duration}s using device: {device}"
    )

    agent = DeepNashAgent(compile=compile_policy).to(device)

    env = make_env(env_name).to(device)
    tensordict = env.reset()

    for _ in range(10):
        tensordict = agent(tensordict)
        tensordict = env.step(tensordict)["next"]
    tensordict = env.reset()

    start_time = perf_counter()
    count = 0
    games = 0
    turns_per_game = []

    while perf_counter() - start_time < duration:
        tensordict = agent(tensordict)
        tensordict = env.step(tensordict)["next"]
        count += 1
        if tensordict["terminated"]:
            info = env.unwrapped.get_info()
            turns = int(info["total_moves"])
            turns_per_game.append(turns)
            games += 1
            logger.info(
                f"Game {games}: Player {-1 * info['cur_player']} received {int(tensordict['reward'].item())}, "
                + f"total_moves: {turns}, moves_since_attack: {int(info['moves_since_attack'])}"
            )
            tensordict = env.reset()

    elapsed = perf_counter() - start_time
    fps = count / elapsed
    avg_game_length = sum(turns_per_game) / len(turns_per_game) if turns_per_game else 0

    logger.info("=== Benchmark Summary ===")
    logger.info(f"Total steps: {count}")
    logger.info(f"Elapsed time: {elapsed:.2f}s")
    logger.info(f"Steps per second (FPS): {fps:.2f}")
    logger.info(f"Games played: {games}")
    logger.info(f"Average game length: {avg_game_length:.2f} turns")

    return {
        "steps": count,
        "games": games,
        "fps": fps,
        "avg_game_length": avg_game_length,
        "duration": elapsed,
    }


def rollout_test(
    env_name, device, compile_policy, duration, max_steps, log_to_file=False
):
    logger = setup_logger(log_to_file)
    logger.info(f"Starting rollout_test with {max_steps} max steps per rollout")
    logger.info(
        f"Using environment: {env_name}, device: {device}, duration: {duration}s"
    )

    agent = DeepNashAgent(compile=compile_policy).to(device)

    env = make_env(env_name).to(device)
    tensordict = env.reset()

    for _ in range(10):
        tensordict = agent(tensordict)
        tensordict = env.step(tensordict)["next"]
    tensordict = env.reset()

    start_time = perf_counter()
    count = 0
    games = 0
    turns_per_game = []

    while perf_counter() - start_time < duration:
        tensordict_rollout = env.rollout(max_steps=max_steps, policy=agent)
        steps = tensordict_rollout.batch_size[0]
        count += steps

        last = tensordict_rollout[-1]["next"]
        if last["terminated"]:
            info = env.unwrapped.get_info()
            turns = info["total_moves"]
            turns_per_game.append(turns)
            games += 1
            logger.info(
                f"Game {games}: Player {-1 * last['cur_player']} received {int(last['reward'].item())}, "
                + f"total_moves: {turns}, moves_since_attack: {info['moves_since_attack']}"
            )

    elapsed = perf_counter() - start_time
    fps = count / elapsed
    avg_game_length = sum(turns_per_game) / len(turns_per_game) if turns_per_game else 0

    logger.info("=== Benchmark Summary ===")
    logger.info(f"Total steps: {count}")
    logger.info(f"Elapsed time: {elapsed:.2f}s")
    logger.info(f"Steps per second (FPS): {fps:.2f}")
    logger.info(f"Games played: {games}")
    logger.info(f"Average game length: {avg_game_length:.2f} turns")

    return {
        "steps": count,
        "games": games,
        "fps": fps,
        "avg_game_length": avg_game_length,
        "duration": elapsed,
    }


def get_directory_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            file_path = os.path.join(dirpath, f)
            total_size += os.path.getsize(file_path)
    return total_size


def vectorized_test(
    env_name,
    device,
    compile_policy,
    duration,
    n_procs,
    n_workers,
    max_steps,
    log_to_file=False,
):
    logger = setup_logger(log_to_file)
    logger.info(
        f"Starting vectorized_test with {n_procs} processes × {n_workers} workers: {max_steps} steps per batch"
    )
    logger.info(
        f"Using environment: {env_name}, device: {device}, duration: {duration}s"
    )

    agent = DeepNashAgent(compile=compile_policy)

    # Trigger lazy initialization
    env = make_env(env_name)
    agent(env.reset())

    from torchrl.data import (
        TensorDictReplayBuffer,
        H5StorageCheckpointer,
        NestedStorageCheckpointer,
        FlatStorageCheckpointer,
    )

    def collate(data):
        print(data)
        # print(data["collector"]["traj_ids"].tolist())
        # print(torch.unique(data["collector"]["traj_ids"]))
        return data[0]

    from torchrl.data.replay_buffers import (
        LazyMemmapStorage,
        LazyMemmapStorage,
        TensorDictReplayBuffer,
        ListStorage,
    )

    memory = ReplayBuffer(
        storage=LazyMemmapStorage(1, ndim=1),
        # sampler=SliceSampler(
        #     slice_len=250, traj_key=("collector", "traj_ids"), strict_length=True
        # ),
        checkpointer=FlatStorageCheckpointer(),
        batch_size=2,
        collate_fn=collate,
        dim_extend=0,
    )

    collector = MultiSyncDataCollector(
        [ParallelEnv(n_workers, partial(make_env, env_name))] * n_procs,
        agent,
        frames_per_batch=3600 * n_workers * n_procs,
        max_frames_per_traj=3600,
        reset_at_each_iter=False,
        total_frames=-1,
        cat_results=0,
        device=device,
        split_trajs=True,
    )

    start_time = perf_counter()
    total_frames = 0
    batches = 0

    for data in collector:
        frames = data.numel()
        print(list(data["collector"].keys()))
        print(data["collector"]["traj_ids"][:, 10])
        print(data["collector"]["traj_ids"][:, -10])

        from tensordict import pad, pad_sequence

        # print()

        mask_lens = data["collector"]["mask"].sum(dim=-1)

        for i, chank in enumerate(data):
            # mask_len = mask_lens[i]
            # chank = chank[:mask_len]
            chank.set("obs", chank["obs"].to(torch.float16))
            chank["next"].set("obs", chank["next"]["obs"].to(torch.float16))

            chank.set("action", chank["action"].to(torch.bool))
            chank.set("action_mask", chank["action_mask"].to(torch.bool))
            chank["next"].set(
                "action_mask", chank["next"]["action_mask"].to(torch.bool)
            )

            chank = pad(chank, [0, 3600 - chank.shape[0]])[None,]
            print(chank)
            print(chank.shape)
            print(chank.batch_size)
            print(memory.extend(chank))

        total_frames += frames
        batches += 1
        logger.info(
            f"Batch {batches} of size {list(data.shape)} added to replay buffer."
        )

        if (perf_counter() - start_time) > duration:
            break

    collector.shutdown()
    del collector

    with tempfile.TemporaryDirectory() as tmpdir:
        memory.dump(tmpdir)
        dir_size = get_directory_size(tmpdir)
        print(f"Size of temporary directory: {dir_size} bytes")

    # data = memory.sample()
    # print(1, data)

    # data = memory.sample()
    # print(2, data)

    # data = memory.sample()
    # print(3, data)

    # data = memory.sample()
    # print(4, data)

    # data = memory.sample()
    # print(5, data)

    elapsed = perf_counter() - start_time
    fps = total_frames / elapsed

    logger.info("=== Benchmark Summary ===")
    logger.info(f"Total frames: {total_frames}")
    logger.info(f"Total batches: {batches}")
    logger.info(f"Elapsed time: {elapsed:.2f}s")
    logger.info(f"Frames per second (FPS): {fps:.2f}")

    return {
        "frames": total_frames,
        "batches": batches,
        "fps": fps,
        "duration": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Stratego environment tests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        usage=argparse.SUPPRESS,
    )

    general = parser.add_argument_group("General settings")
    general.add_argument(
        "--test",
        choices=["basic", "policy", "rollout", "vectorized"],
        default="policy",
        help="Which test to run.",
    )
    general.add_argument(
        "--env_impl",
        choices=["python", "cpp"],
        default="cpp",
        help="Select environment implementation.",
    )
    general.add_argument(
        "--device", type=str, default="cuda", help="Device to use (e.g., 'cuda', 'cpu')"
    )
    general.add_argument(
        "--duration", type=int, default=60, help="Test duration in seconds."
    )
    general.add_argument(
        "--max_steps",
        type=int,
        default=2000,
        help="Max number of steps in rollout/vectorized test.",
    )

    vectorized = parser.add_argument_group("Vectorized test settings")
    vectorized.add_argument(
        "--n_procs", type=int, default=4, help="Number of processes."
    )
    vectorized.add_argument(
        "--n_workers", type=int, default=4, help="Number of workers per process."
    )

    # Флаги
    flags = parser.add_argument_group("Flags")
    flags.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for policy network.",
    )

    args = parser.parse_args()

    env_name = (
        "stratego_gym/Stratego-v0"
        if args.env_impl == "python"
        else "stratego_gym/StrategoCpp-v0"
    )

    if args.test == "basic":
        basic_test(env_name, args.duration)
    elif args.test == "policy":
        policy_test(env_name, args.device, args.compile, args.duration)
    elif args.test == "rollout":
        rollout_test(env_name, args.device, args.compile, args.duration, args.max_steps)
    elif args.test == "vectorized":
        vectorized_test(
            env_name,
            args.device,
            args.compile,
            args.duration,
            args.n_procs,
            args.n_workers,
            args.max_steps,
        )


if __name__ == "__main__":
    main()
