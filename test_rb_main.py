import argparse
from functools import partial
import logging
import os
import tempfile
from time import perf_counter

import gymnasium as gym
import tensordict
import torch
from torchrl.collectors import (
    MultiSyncDataCollector,
    MultiaSyncDataCollector,
    SyncDataCollector,
)
from torchrl.data import LazyTensorStorage, SliceSampler, ReplayBuffer
from torchrl.envs import GymEnv, TrajCounter
from torchrl.envs import default_info_dict_reader, ParallelEnv, SerialEnv

from deepnash import DeepNashAgent
from deepnash.resources.replay_buffer import CustomReplayBufferEnsemble
from deepnash.resources.transforms import QuantizeTransform


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


def get_directory_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            file_path = os.path.join(dirpath, f)
            total_size += os.path.getsize(file_path)
    return total_size


def make_env(env_name):
    reader = default_info_dict_reader(["cur_player"])
    return (
        GymEnv(env_name, render_mode=None).set_info_dict_reader(reader)
        # .append_transform(TrajCounter())
    )


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

    memory = CustomReplayBufferEnsemble(
        num_buffer_sampled=1,
        buffer_size=10,
        batch_size=2,
    )

    collector = SyncDataCollector(
        # [ParallelEnv(n_workers, partial(make_env, env_name))] * n_procs,
        # [partial(make_env, env_name) for _ in range(n_workers)],
        ParallelEnv(n_workers * n_procs, partial(make_env, env_name)).append_transform(
            TrajCounter()
        ),
        # env,
        agent,
        frames_per_batch=3600 * 2,
        max_frames_per_traj=3600,
        # reset_at_each_iter=False,
        total_frames=-1,
        # cat_results="stack",
        device=device,
        # policy_device=device,
        # env_device="cpu",
        # storing_device=device,
        split_trajs=True,
        # postproc=QuantizeTransform(
        #     in_keys=["obs", ("next", "obs")],
        #     out_keys=["obs", ("next", "obs")],
        # ),
        reset_at_each_iter=False,
    )

    start_time = perf_counter()
    total_frames = 0
    batches = 0

    for data in collector:

        data.set("obs", data["obs"].to(torch.float16))
        data["next"].set("obs", data["next"]["obs"].to(torch.float16))
        data.set("action", data["action"].to(torch.bool))
        data.set("action_mask", data["action_mask"].to(torch.bool))
        data["next"].set("action_mask", data["next"]["action_mask"].to(torch.bool))
        # del data["obs"]
        # del data["next", "obs"]

        data.set("priority", data["value"])

        # print(data)

        total_frames_added, games_added, games_stashed, games_recovered = (
            memory.extend_batch(data)
        )
        total_frames += total_frames_added

        batches += 1
        logger.info(
            f"Batch {batches} of size {list(data.shape)} added to replay buffer."
        )

        if (perf_counter() - start_time) > duration:
            break

    collector.shutdown()
    del collector

    for i in range(18):
        print(f"Buffer {(i + 1) * 200} len:", len(memory[i]))

    print(memory[1])
    print(memory[1].sample(return_info=True))

    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = os.path.join(tmpdir, "storage")
        os.makedirs(storage_path, exist_ok=True)

        memory._rbs[0].dumps(tmpdir)
        dir_size = get_directory_size(tmpdir)
        print(f"Size of temporary directory: {dir_size} bytes")

        # Печатаем содержимое директории storage
        print(f"Содержимое папки {storage_path}:")
        print(os.listdir(storage_path))

    elapsed = perf_counter() - start_time
    fps = total_frames / elapsed

    logger.info("=== Benchmark Summary ===")
    logger.info(f"Total frames: {total_frames}")
    logger.info(f"Total batches: {batches}")
    logger.info(f"Elapsed time: {elapsed:.2f}s")
    logger.info(f"Frames per second (FPS): {fps:.2f}")


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
        "--n_procs", type=int, default=2, help="Number of processes."
    )
    vectorized.add_argument(
        "--n_workers", type=int, default=1, help="Number of workers per process."
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
