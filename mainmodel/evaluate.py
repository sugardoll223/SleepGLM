from __future__ import annotations

import argparse

from mainmodel.train import run_training
from mainmodel.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model evaluation entrypoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config.")
    parser.add_argument("--checkpoint", type=str, required=False, default="", help="Optional checkpoint path.")
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        default=[],
        help="Override config with dotted key format, e.g. data.test_files=[\"/path/test.h5\"]",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    args.eval_only = True
    run_training(cfg, args)


if __name__ == "__main__":
    main()


