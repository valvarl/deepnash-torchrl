import argparse
import torch
from torchinfo import summary

from deepnash.agents.stratego import StrategoAgent


def model_benchmark(inner_channels, outer_channels, N, M, device):
    model = DeepNashNet(inner_channels, outer_channels, N, M).to(device)

    example_input = torch.randn(1, 82, 10, 10).to(device)
    example_input[:, -3:-2, :, :] = 1
    example_input[:, -2:-1, :, :] = 0
    example_mask = torch.ones(1, 10, 10, dtype=torch.bool).to(device)

    with torch.no_grad():
        output = model(example_input, example_mask)

    print("Model forward pass successful.\n")

    summary(
        model,
        input_data=(example_input, example_mask),
        col_names=["input_size", "output_size", "num_params"],
        depth=3,
    )


def main():
    parser = argparse.ArgumentParser(
        description="DeepNashNet Model Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        usage=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--inner_channels",
        type=int,
        default=320,
        help="Number of inner channels in the network",
    )
    parser.add_argument(
        "--outer_channels",
        type=int,
        default=256,
        help="Number of outer channels in the network",
    )
    parser.add_argument(
        "-N",
        type=int,
        default=2,
        help="Number of residual blocks in each inner tower (N)",
    )
    parser.add_argument(
        "-M", type=int, default=2, help="Number of outer residual blocks (M)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run the benchmark on",
    )

    args = parser.parse_args()

    model_benchmark(
        inner_channels=args.inner_channels,
        outer_channels=args.outer_channels,
        N=args.N,
        M=args.M,
        device=args.device,
    )


if __name__ == "__main__":
    main()
