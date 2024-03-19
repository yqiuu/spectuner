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


def main(config):
    T_back = config["sl_model"].get("tBack", 0.)
    obs_data = load_preprocess(config["file_spec"], T_back)
    prominence = config["opt_single"]["pm_loss"]["prominence"]
    rel_height =  config["opt_single"]["pm_loss"]["rel_height"]
    idn = Identification(obs_data, T_back, prominence, rel_height)
    dirname = Path(config["save_dir"])/Path(config["opt_single"]["dirname"])

    fname_base = config.get("fname_base", None)
    if fname_base is None:
        res = identify_without_base(idn, dirname, config)
    else:
        res = identify_with_base(idn, dirname, config)
    pickle.dump(res, open(dirname/Path("identify_result.pickle"), "wb"))


def identify_without_base(idn, dirname, config):
    res_list = []
    for fname in dirname.glob("*.pickle"):
        if fname.name == "identify_result.pickle":
            continue
        data = pickle.load(open(fname, "rb"))
        res = idn.identify(
            data["mol_store"], config["sl_model"],
            data["params_best"], data["T_pred"]
        )
        if len(res.df_mol) == 0:
            continue
        res = res.extract_sub(data["mol_store"].mol_list[0]["id"])
        res_list.append(res)
    return concat_identify_result(res_list)


def identify_with_base(idn, dirname, config):
    config_slm = config["sl_model"]

    data = pickle.load(open(config["fname_base"], "rb"))
    mol_store_base = data["mol_store"]
    params_base = data["params_best"]

    res_list = []
    for fname in dirname.glob("*.pickle"):
        if fname.name == "identify_result.pickle":
            continue
        data = pickle.load(open(fname, "rb"))
        mol_store_combine, params_combine = combine_mol_stores(
            [mol_store_base, data["mol_store"]],
            [params_base, data["params_best"]],
            config["sl_model"]
        )
        T_pred_data = mol_store_combine.compute_T_pred_data(
            params_combine, data["freq"], config_slm
        )
        res = idn.identify(
            mol_store_combine, config_slm, params_combine, T_pred_data
        )
        res = res.extract_sub(data["mol_store"].mol_list[0]["id"])
        res_list.append(res)
    return concat_identify_result(res_list)


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    main(config)