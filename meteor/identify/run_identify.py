import pickle
from pathlib import Path

from .identify import identify_file, identify_without_base, identify_with_base
from ..preprocess import load_preprocess
from ..algorithms import Identification


__all__ = ["run_identify"]


def run_identify(config, target):
    T_back = config["sl_model"].get("tBack", 0.)
    obs_data = load_preprocess(config["file_spec"], T_back)
    prominence = config["opt_single"]["pm_loss"]["prominence"]
    rel_height =  config["opt_single"]["pm_loss"]["rel_height"]
    idn = Identification(obs_data, T_back, prominence, rel_height)

    if target == "single":
        key = "opt_single"
        fname_base = config.get("fname_base", None)
    elif target == "combine":
        key = "opt_combine"
        fname_base =  Path(config["save_dir"]) \
            / Path(config["opt_combine"]["dirname"]) \
            / Path("combine.pickle")
    else:
        target = Path(target)
        if target.exists():
            res = identify_file(idn, target, config)
            pickle.dump(res, open(_rename(target), "wb"))
            return
        else:
            raise ValueError(f"Unknown target: {target}.")

    dirname = Path(config["save_dir"])/Path(config[key]["dirname"])
    if fname_base is None:
        res = identify_without_base(idn, dirname, config)
    else:
        res = identify_with_base(idn, dirname, fname_base, config)
    pickle.dump(res, open(dirname/Path("identify.pickle"), "wb"))


def _rename(fname):
    return fname.parent/f"identify_{fname.name}"