"""Stable console interface between Seg4D and the Point Transformer V3 runtime."""

import argparse
import os
import sys


if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from point_transformer_v3 import inference, training
else:
    from . import inference, training


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if arguments and arguments[0] == "train":
        training.main(arguments[1:])
        return 0
    if arguments and arguments[0] == "predict":
        inference.main(arguments[1:])
        return 0

    parser = argparse.ArgumentParser(
        prog="point_transformer_v3",
        description="Seg4D Point Transformer V3 console runtime.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Load the CUDA model and run a small forward pass.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="Command to run: train or predict.",
    )
    known = parser.parse_args(arguments)

    if known.self_test:
        if known.command:
            parser.error("--self-test cannot be combined with another command.")
        training.self_test()
        return 0
    if known.command:
        parser.error(f"unknown command: {known.command}")

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
