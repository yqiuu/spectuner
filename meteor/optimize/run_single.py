import pickle
from pathlib import Path

import numpy as np

from .optimize import load_base_data, optimize, create_pool
from ..config import append_freqs_exclude
from ..preprocess import load_preprocess, get_freq_data
from ..spectral_data import query_molecules
from ..xclass_wrapper import MoleculeStore
from ..identify import PeakManager
from ..identify.identify import identify
from ..fitting_model import create_fitting_model


__all__ = ["run_single", "select_molecules"]


def run_single(config, parent_dir, need_identify=True):
    fname_base = config.get("fname_base", None)
    T_base_data, freqs, spans, id_offset = load_base_data(fname_base)
    config = append_freqs_exclude(config, freqs)

    obs_data = load_preprocess_from_config(config)
    mol_list, include_dict = select_molecules(obs_data, spans, config)

    save_dir = Path(parent_dir)/"single"
    save_dir.mkdir(exist_ok=True)

    with create_pool(config["opt"]["n_process"], config["opt"]["use_mpi"]) as pool:
        for item in mol_list:
            name = item["root"]
            item["id"] = item["id"] + id_offset
            mol_store = MoleculeStore([item], include_dict[name])
            model = create_fitting_model(
                obs_data, mol_store, config, T_base_data
            )
            ret_dict = optimize(model, config["opt"], pool)
            pickle.dump(ret_dict, open(save_dir/Path("{}.pickle".format(name)), "wb"))

        if need_identify:
            identify(config, parent_dir, "single")


def select_molecules(obs_data, spans, config):
    if len(spans) == 0:
        T_back = config["sl_model"].get("tBack", 0.)
        peak_mgr = PeakManager(
            obs_data, T_back, **config["peak_manager"]
        )
        freqs = np.mean(np.vstack(peak_mgr.spans_obs_data), axis=1)
    else:
        freqs = np.mean(spans, axis=1)
    mol_list, include_dict = query_molecules(
        get_freq_data(obs_data),
        v_LSR=config["sl_model"].get("vLSR", 0.),
        freqs_include=freqs,
        v_range=config["opt"]["bounds"]["v_LSR"],
        **config["species"],
    )
    return mol_list, include_dict


def load_preprocess_from_config(config):
    file_spec = config["files"]
    T_back = config["sl_model"].get("tBack", 0.)
    return load_preprocess(file_spec, T_back)