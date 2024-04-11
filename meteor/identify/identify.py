import pickle
from copy import deepcopy
from pathlib import Path

from .ident_result import compute_T_single_data
from .peaks import PeakManager
from ..utils import load_pred_data
from ..xclass_wrapper import combine_mol_stores
from ..preprocess import load_preprocess


def identify(config, parent_dir, target):
    T_back = config["sl_model"].get("tBack", 0.)
    obs_data = load_preprocess(config["files"], T_back)
    prominence = config["pm_loss"]["prominence"]
    rel_height =  config["pm_loss"]["rel_height"]
    idn = PeakManager(obs_data, T_back, prominence, rel_height)

    if target == "single":
        fname_base = config.get("fname_base", None)
    elif target == "combine":
        fname_base =  Path(parent_dir)/"combine"/"combine.pickle"
    else:
        # The function will always terminate in this block
        target = Path(target)
        if target.exists():
            res = identify_file(idn, target, config)
            save_name = target.parent/f"identify_{target.name}"
            pickle.dump(res, open(save_name, "wb"))
            return
        else:
            raise ValueError(f"Unknown target: {target}.")

    dirname = Path(parent_dir)/target
    if fname_base is None:
        res = identify_without_base(idn, dirname, config)
    else:
        res = identify_with_base(idn, dirname, fname_base, config)
    pickle.dump(res, open(dirname/Path("identify.pickle"), "wb"))


def identify_file(idn, fname, config):
    data = pickle.load(open(fname, "rb"))
    res = idn.identify(
        data["mol_store"], config["sl_model"], data["params_best"],
    )
    return res


def identify_without_base(idn, dirname, config):
    pred_data_list = load_pred_data(dirname.glob("*.pickle"), reset_id=True)
    res_dict = {}
    for data in pred_data_list:
        assert len(data["mol_store"].mol_list) == 1
        key = data["mol_store"].mol_list[0]["id"]
        res = idn.identify(
            data["mol_store"], config["sl_model"], data["params_best"],
        )
        if len(res.df_mol) == 0:
            res = None
        res_dict[key] = res
    return res_dict


def identify_with_base(idn, dirname, fname_base, config):
    config_slm = config["sl_model"]

    data = pickle.load(open(fname_base, "rb"))
    mol_store_base = data["mol_store"]
    params_base = data["params_best"]
    T_single_dict_base = compute_T_single_data(
        mol_store_base, config_slm, params_base, data["freq"]
    )

    pred_data_list = load_pred_data(dirname.glob("*.pickle"), reset_id=True)
    res_dict = {}
    for data in pred_data_list:
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
        if res.is_empty() == 0:
            continue
        assert len(data["mol_store"].mol_list) == 1
        key = data["mol_store"].mol_list[0]["id"]
        try:
            res = res.extract(key)
        except KeyError:
            res = None
        res_dict[key] = res
    return res_dict

