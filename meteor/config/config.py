import yaml
import shutil
from pathlib import Path


__all__ = ["create_config", "load_config"]


def create_config(dir="./"):
    template_dir = Path(__file__).resolve().parent/Path("templates")
    target_dir = Path(dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(template_dir/"config.yml", target_dir)
    for _, fname in iter_config_names():
        shutil.copy(template_dir/fname, target_dir)

    tmp_dir = target_dir/'tmp'
    tmp_dir.mkdir(exist_ok=True)


def load_config(dir):
    dir = Path(dir)
    config = yaml.safe_load(open(dir/"config.yml"))
    if config["files"] is None or len(config["files"]) == 0:
        raise ValueError("'files' cannot be empty.")
    if config["pm_loss"]["prominence"] is None:
        raise ValueError("'prominence' cannot be None.")
    if config["pm_loss"]["rel_height"] is None:
        raise ValueError("'rel_height' cannot be None.")
    for key, fname in iter_config_names():
        config[key] = yaml.safe_load(open(fname))
    return config


def iter_config_names():
    keys = ["opt_single", "opt_combine", "species", "modify"]
    file_names = [
        "config_opt_single.yml",
        "config_opt_combine.yml",
        "species.yml",
        "modify.yml"
    ]
    return zip(keys, file_names)