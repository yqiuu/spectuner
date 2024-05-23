import pickle
from pathlib import Path
from copy import deepcopy

import numpy as np

from .optimize import optimize, create_pool
from ..preprocess import load_preprocess, get_freq_data
from ..spectral_data import select_molecules
from ..xclass_wrapper import MoleculeStore
from ..identify.identify import identify
from ..fitting_model import create_fitting_model


__all__ = ["run_single"]


def run_single(config, parent_dir, need_identify=True):
    id_offset = 0
    fname_base = config.get("fname_base", None)
    if fname_base is not None:
        res = pickle.load(open(fname_base, "rb"))
        base_data = res.get_T_pred()
        freqs = res.get_unknown_lines()
        spans = derive_unknown_spans(freqs, *config["opt"]["bounds"]["v_LSR"])
        config_species = deepcopy(config["species"])
        if config_species["exclude_list"] is None:
            config_species["exclude_list"] = []
        config_species["exclude_list"].extend(derive_exclude_list(res))
        for key in res.mol_data:
            id_offset = max(id_offset, key)
        id_offset += 1
    else:
        base_data = None
        freqs = None
        spans = None
        config_species = config["species"]

    save_dir = Path(parent_dir)/"single"
    save_dir.mkdir(exist_ok=True)

    obs_data = load_preprocess_from_config(config)
    mol_list, include_dict = select_molecules(
        get_freq_data(obs_data), spans=spans, **config_species,
    )
    print(mol_list)
    print(len(mol_list))

    with create_pool(config["opt"]["n_process"], config["opt"]["use_mpi"]) as pool:
        for item in mol_list:
            name = item["root"]
            item["id"] = item["id"] + id_offset
            mol_store = MoleculeStore([item], include_dict[name])
            model = create_fitting_model(
                obs_data, mol_store, config, config["opt"], base_data, freqs
            )
            ret_dict = optimize(model, config["opt"], pool)
            pickle.dump(ret_dict, open(save_dir/Path("{}.pickle".format(name)), "wb"))

        if need_identify:
            identify(config, parent_dir, "single")


def load_preprocess_from_config(config):
    file_spec = config["files"]
    T_back = config["sl_model"].get("tBack", 0.)
    return load_preprocess(file_spec, T_back)


def derive_unknown_spans(freqs, v_min, v_max):
    c = 3e5 # speed of light [km/s]
    freqs_min = freqs*(1 + v_min/c)
    freqs_max = freqs*(1 + v_max/c)
    return np.vstack([freqs_min, freqs_max]).T


def derive_exclude_list(res):
    exclude_list = []
    for sub_dict in res.mol_data.values():
        for key in sub_dict:
            exclude_list.append(key)
    return exclude_list