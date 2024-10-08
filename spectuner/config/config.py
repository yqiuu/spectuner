import yaml
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np


__all__ = ["create_config", "load_config", "append_exclude_info"]


def create_config(dir="./"):
    template_dir = Path(__file__).resolve().parent/Path("templates")
    target_dir = Path(dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = target_dir.resolve()/'tmp'
    tmp_dir.mkdir(exist_ok=True)
    lines = open(template_dir/"config.yml").readlines()
    for i_l, ln in enumerate(lines):
        if "TMP_DIR" in ln:
            lines[i_l] = ln.replace("TMP_DIR", str(tmp_dir/"tmp"))
            break
    open(target_dir/"config.yml", "w").writelines(lines)

    for _, fname in iter_config_names():
        shutil.copy(template_dir/fname, target_dir)

    tmp_dir = target_dir/'tmp'
    tmp_dir.mkdir(exist_ok=True)


def load_config(dir):
    dir = Path(dir)
    config = yaml.safe_load(open(dir/"config.yml"))
    if config["files"] is None or len(config["files"]) == 0:
        raise ValueError("'files' cannot be empty.")
    if config["peak_manager"]["prominence"] is None:
        raise ValueError("'prominence' cannot be None.")
    if config["peak_manager"]["rel_height"] is None:
        raise ValueError("'rel_height' cannot be None.")
    for key, fname in iter_config_names():
        config[key] = yaml.safe_load(open(dir/fname))
    if config["peak_manager"]["freqs_exclude"] is None:
        config["peak_manager"]["freqs_exclude"] = np.zeros(0)
    else:
        config["peak_manager"]["freqs_exclude"] \
            = np.loadtxt(config["peak_manager"]["freqs_exclude"])
    return config


def iter_config_names():
    keys = ["species", "modify"]
    file_names = [
        "species.yml",
        "modify.yml"
    ]
    return zip(keys, file_names)


def append_exclude_info(config, freqs_exclude, exlude_list):
    config = deepcopy(config)
    if config["peak_manager"]["freqs_exclude"] is None:
        config["peak_manager"]["freqs_exclude"] = np.zeros(0)
    config["peak_manager"]["freqs_exclude"] \
        = np.append(config["peak_manager"]["freqs_exclude"], freqs_exclude)
    if config["species"]["exclude_list"] is None:
        config["species"]["exclude_list"] = []
    config["species"]["exclude_list"].extend(exlude_list)
    return config