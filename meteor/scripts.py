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
    parser.add_argument("--target", type=str, default="full")
    parser.add_argument("--new_cycle", action="store_true", default=False)
    args = parser.parse_args()

    config = load_config(args.config)
    idx = get_lst_dir_index(args.config)
    if args.new_cycle or idx == -1:
        idx += 1
    # Set fname_base
    if idx > 0:
        fname_base = Path(f"cycle_{idx - 1}")/"combine"/"combine_final.pickle"
        print(fname_base)
        if not fname_base.exists():
            raise ValueError("Finalize the last cycle before staring a new one.")
        config["fname_base"] = fname_base

    save_dir = Path(f"cycle_{idx}")
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.target == "single":
        run_single(config, save_dir)
    elif args.target == "combine":
        run_combine(config, save_dir)
    elif args.target == "full":
        run_single(config, save_dir, need_identify=False)
        run_combine(config, save_dir)
    else:
        raise ValueError(f"Unknown target: {args.target}.")


def exec_modify():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    idx = get_lst_dir_index(args.config)
    dirname = Path(args.config)/f"cycle_{idx}"
    modify(config, dirname)


def exec_identify():
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("dir", type=str)
    parser.add_argument("target", type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    identify(config, args.dir, args.target)


def get_lst_dir_index(dirname):
    inds = []
    for fname in Path(dirname).glob("cycle_*"):
        match = re.search(r'cycle_(\d+)', fname.name)
        inds.append(int(match.group(1)))
    if len(inds) == 0:
        return -1
    return max(inds)