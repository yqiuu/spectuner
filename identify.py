import sys
import yaml
import pickle
from pathlib import Path
from copy import deepcopy

from src.preprocess import load_preprocess
from src.algorithms import (
    concat_identify_result, compute_T_single_data, Identification
)
from src.optimize import combine_mol_stores


def main(config, target):
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
        raise ValueError()

    dirname = Path(config["save_dir"])/Path(config[key]["dirname"])
    if fname_base is None:
        res = identify_without_base(idn, dirname, config)
    else:
        res = identify_with_base(idn, dirname, fname_base, config)
    print(res)
    pickle.dump(res, open(dirname/Path("identify.pickle"), "wb"))


def identify_without_base(idn, dirname, config):
    res_list = []
    for fname in dirname.glob("*.pickle"):
        if is_exclusive(fname):
            continue
        data = pickle.load(open(fname, "rb"))
        res = idn.identify(
            data["mol_store"], config["sl_model"],
            data["params_best"],
        )
        if len(res.df_mol) == 0:
            continue
        res = res.extract_sub(data["mol_store"].mol_list[0]["id"])
        res_list.append(res)
    return concat_identify_result(res_list)


def identify_with_base(idn, dirname, fname_base, config):
    config_slm = config["sl_model"]

    data = pickle.load(open(fname_base, "rb"))
    mol_store_base = data["mol_store"]
    params_base = data["params_best"]
    T_single_dict_base = compute_T_single_data(
        mol_store_base, config_slm, params_base, data["freq"]
    )

    res_list = []
    for fname in dirname.glob("*.pickle"):
        if is_exclusive(fname):
            continue
        data = pickle.load(open(fname, "rb"))
        mol_store_combine, params_combine = combine_mol_stores(
            [mol_store_base, data["mol_store"]],
            [params_base, data["params_best"]],
            config["sl_model"]
        )
        T_single_dict = deepcopy(T_single_dict_base)
        T_single_dict.update(compute_T_single_data(
            data["mol_store"], config_slm, data["params_best"], data["freq"]
        ))
        res = idn.identify(
            mol_store_combine, config_slm, params_combine, T_single_dict
        )
        if len(res.df_mol) == 0:
            continue
        try:
            res = res.extract_sub(data["mol_store"].mol_list[0]["id"])
        except KeyError:
            res = None
        if res is not None:
            res_list.append(res)
    return concat_identify_result(res_list)


def is_exclusive(fname):
    name = str(fname.name)
    return name.startswith("identify") \
        or name.startswith("combine") \
        or name.startswith("tmp")


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    target = sys.argv[2]
    main(config, target)