import re
from pathlib import Path
from argparse import ArgumentParser

from .config import create_config, load_config
from .optimize import run_single, run_combine, modify
from .identify.identify import identify


def exec_config():
    parser = ArgumentParser()
    parser.add_argument("dir", type=str)
    args = parser.parse_args()
    create_config(args.dir)


def exec_fit():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument("--mode", type=str, default="entire")
    parser.add_argument("--fbase", type=str, default="")
    args = parser.parse_args()

    config = load_config(args.config)
    # Set fname_base
    if args.fbase != "":
        config["fname_base"] = args.fbase

    save_dir = Path(args.target)
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "single":
        run_single(config, save_dir)
    elif args.mode == "combine":
        run_combine(config, save_dir)
    elif args.mode == "entire":
        run_single(config, save_dir, need_identify=False)
        run_combine(config, save_dir)
    else:
        raise ValueError(f"Unknown mode: {args.mode}.")


def exec_modify():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("target", type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    modify(config, args.target)


def exec_identify():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument("--mode", type=str, default="single")
    args = parser.parse_args()

    config = load_config(args.config)
    identify(config, args.target, args.mode)


def get_lst_dir_index(dirname):
    inds = []
    for fname in Path(dirname).glob("cycle_*"):
        match = re.search(r'cycle_(\d+)', fname.name)
        inds.append(int(match.group(1)))
    if len(inds) == 0:
        return -1
    return max(inds)