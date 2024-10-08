import pickle
from pathlib import Path

import numpy as np

from .optimize import prepare_base_props, optimize, create_pool
from ..config import append_exclude_info
from ..preprocess import load_preprocess, get_freq_data
from ..spectral_data import query_molecules
from ..xclass_wrapper import MoleculeStore
from ..identify import PeakManager
from ..identify.identify import identify
from ..fitting_model import create_fitting_model


__all__ = ["run_single", "select_molecules"]


def run_single(config, result_dir, need_identify=True):
    """Run the individual fitting phase.

    Args:
        config (dict): Config.
        result_dir (str): Directory to save the results.
        need_identify (bool): If ``True``, peform the identification.
    """
    fname_base = config.get("fname_base", None)
    base_props = prepare_base_props(fname_base, config)
    config = append_exclude_info(
        config, base_props["freqs_exclude"], base_props["exclude_list"]
    )

    obs_data = load_preprocess_from_config(config)
    mol_list, include_dict = select_molecules(
        obs_data, base_props["spans_include"], config
    )

    save_dir = Path(result_dir)/"single"
    save_dir.mkdir(exist_ok=True)

    use_mpi = config["opt"].get("use_mpi", False)
    with create_pool(config["opt"]["n_process"], use_mpi) as pool:
        for item in mol_list:
            name = item["root"]
            item["id"] = item["id"] + base_props["id_offset"]
            mol_store = MoleculeStore([item], include_dict[name])
            model = create_fitting_model(
                obs_data, mol_store, config, base_props["T_base"]
            )
            ret_dict = optimize(model, config["opt"], pool)
            pickle.dump(ret_dict, open(save_dir/Path("{}.pickle".format(name)), "wb"))

        if need_identify:
            identify(config, result_dir, "single")


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