import pickle
from copy import deepcopy
from pathlib import Path

from .peaks import PeakManager
from .. import sl_model
from ..utils import load_pred_data
from ..xclass_wrapper import combine_mol_stores
from ..preprocess import load_preprocess


def identify(config, target, mode=None):
    """Perform identification.

    Args:
        config (dict): Config.
        target (str):
            - If ``target`` is a file, perfrom identification for the target
              file.
            - If ``target`` is a directory, perform identification for all files
              in the directory.

        mode (str): This is only applicable when ``target`` is a directory.
            - ``single``: Use ``fname_base`` given in in the config as base
            data.
            - ``combine``: Use ``combine.pickle`` in the target directory
            as base data.
    """
    T_back = config["sl_model"].get("tBack", 0.)
    files_spec = [item["fname"] for item in config["obs_info"]]
    obs_data = load_preprocess(files_spec, T_back)
    idn = PeakManager(obs_data, T_back, **config["peak_manager"])

    target = Path(target)
    if target.is_file():
        res = identify_file(idn, target, config)
        save_name = target.parent/f"identify_{target.name}"
        pickle.dump(res, open(save_name, "wb"))
    elif target.is_dir():
        if mode == "single":
            fname_base = config.get("fname_base", None)
        elif mode == "combine":
            fname_base = target/"combine"/"combine.pickle"
        else:
            raise ValueError(f"Unknown mode: {mode}.")

        dirname = target/mode
        if fname_base is None:
            res = identify_without_base(idn, dirname, config)
        else:
            res = identify_with_base(idn, dirname, fname_base, config)
        pickle.dump(res, open(dirname/Path("identify.pickle"), "wb"))
    else:
        raise ValueError(f"Unknown target: {target}.")


def identify_file(idn, fname, config):
    data = pickle.load(open(fname, "rb"))
    res = idn.identify(data["specie"], config, data["params_best"])
    return res


def identify_without_base(idn, dirname, config):
    pred_data_list = load_pred_data(dirname.glob("*.pickle"), reset_id=True)
    res_dict = {}
    for data in pred_data_list:
        assert len(data["specie"]) == 1
        key = data["specie"][0]["id"]
        res = idn.identify(data["specie"], config, data["params_best"])
        if res.is_empty():
            res = None
        res_dict[key] = res
    return res_dict


def identify_with_base(idn, dirname, fname_base, config):
    data = pickle.load(open(fname_base, "rb"))
    mol_store_base = data["mol_store"]
    params_base = data["params_best"]
    T_single_dict_base = sl_model.compute_T_single_data(
        mol_store_base, config, params_base, data["freq"]
    )

    pred_data_list = load_pred_data(dirname.glob("*.pickle"), reset_id=False)
    res_dict = {}
    for data in pred_data_list:
        mol_store_combine, params_combine = combine_mol_stores(
            [mol_store_base, data["mol_store"]],
            [params_base, data["params_best"]],
        )
        T_single_dict = deepcopy(T_single_dict_base)
        T_single_dict.update(sl_model.compute_T_single_data(
            data["mol_store"], config, data["params_best"], data["freq"]
        ))
        res = idn.identify(
            mol_store_combine, config, params_combine, T_single_dict
        )
        if res.is_empty():
            continue
        assert len(data["mol_store"].mol_list) == 1
        key = data["mol_store"].mol_list[0]["id"]
        try:
            res = res.extract(key)
        except KeyError:
            res = None
        res_dict[key] = res
    return res_dict

