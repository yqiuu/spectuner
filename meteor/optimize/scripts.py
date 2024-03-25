import yaml
from argparse import ArgumentParser

from .run_single import run_single
from .run_combine import run_combine


def main():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--target", type=str, default="full")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    if args.target == "single":
        run_single(config)
    elif args.target == "combine":
        run_combine(config)
    elif args.target == "full":
        run_single(config, need_identify=False)
        run_combine(config)
    else:
        raise ValueError(f"Unknown target: {args.target}.")