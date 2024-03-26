import yaml
from argparse import ArgumentParser

from .config import create_config, load_config
from .optimize import run_single, run_combine, modify
from .identify.identify import identify


def exec_config():
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str, default="./")
    args = parser.parse_args()
    create_config(args.dir)


def exec_fit():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--target", type=str, default="full")
    args = parser.parse_args()

    config = load_config(args.config)
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

    config = load_config(args.config)
    config_modify = config["modify"]
    modify(config, config_modify)


def exec_identify():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("target", type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    identify(config, args.target)