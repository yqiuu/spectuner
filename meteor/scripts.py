import yaml
from argparse import ArgumentParser

from .optimize import run_single, run_combine, modify
from .identify.identify import identify


def exec_fit():
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


def exec_modify():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("config_modify", type=str)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    config_modify = yaml.safe_load(open(args.config_modify))
    modify(config, config_modify)


def exec_identify():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("target", type=str)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    identify(config, args.target)