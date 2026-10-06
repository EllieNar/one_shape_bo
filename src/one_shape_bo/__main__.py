"""Command-line entry point for a package BO run."""

from __future__ import annotations

import argparse

from .optimization import OptimizationConfig, run_optimization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=("triangle", "ellipse"), default="triangle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--burn-in", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument(
        "--local-start-strategy",
        choices=("radius", "archive"),
        default="radius",
    )
    parser.add_argument("--output-dir")
    arguments = parser.parse_args()
    run_optimization(
        OptimizationConfig(
            htype=0 if arguments.shape == "triangle" else 1,
            seed=arguments.seed,
            burn_in=arguments.burn_in,
            iterations=arguments.iterations,
            local_start_strategy=arguments.local_start_strategy,
            output_dir=arguments.output_dir,
        )
    )


if __name__ == "__main__":
    main()
