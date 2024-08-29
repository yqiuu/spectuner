import re
from pathlib import Path
from argparse import ArgumentParser

from .config import create_config, load_config
from .optimize import run_single, run_combine
from .identify import identify
from .modify import modify


def exec_config():
    parser = ArgumentParser(description="Create config files.")
    parser.add_argument(
        "dir", type=str,
        help="Directory to create config files."
    )
    args = parser.parse_args()
    create_config(args.dir)


def exec_fit():
    parser = ArgumentParser(description="Run spectral fitting.")
    parser.add_argument(
        "config", type=str,
        help="Directory to config files."
    )
    parser.add_argument(
        "target", type=str,
        help="Directory to save results."
    )
    parser.add_argument(
        "--mode", type=str, default="entire",
        choices=("single", "combine", "entire"),
        help="""1. single: Run individual fitting phase.
                2. combine: Run combining phase.
                3. entire: Run both phases."""
    )
    parser.add_argument(
        "--fbase", type=str, default="",
        help="File name to a previous combined result (combine.pickle)."
    )
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
    parser = ArgumentParser(description="Modify a combined result.")
    parser.add_argument(
        "config", type=str,
        help="Directory to config files."
    )
    parser.add_argument(
        "target", type=str,
        help="Directory to saved results."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    modify(config, args.target)


def exec_identify():
    parser = ArgumentParser(description="Peform identification.")
    parser.add_argument(
        "config", type=str,
        help="Directory to config files."
    )
    parser.add_argument(
        "target", type=str,
        help="Directory to saved results or File name a fitting result."
    )
    parser.add_argument(
        "--mode", type=str, default="single",
        choices=("single", "combine"),
        help="""1. single: Identify results in the individual fitting phase.
                2. combine: Identify results in the combining phase."""
    )
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