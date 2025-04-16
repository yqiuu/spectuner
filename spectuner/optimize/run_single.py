from pathlib import Path

import h5py
import numpy as np

from .optimize import prepare_base_props, optimize, create_pool, print_fitting
from ..config import append_exclude_info
from ..preprocess import load_preprocess, get_freq_data
from ..sl_model import query_species, SpectralLineDatabase, SpectralLineModelFactory
from ..slm_factory import jit_fitting_model, FittingModel
from ..peaks import PeakManager
from ..identify import identify
from ..utils import save_fitting_result, derive_specie_save_name


__all__ = ["run_single", "create_specie_list"]


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
    obs_data = load_preprocess(config["obs_info"])
    freq_list = get_freq_data(obs_data)
    sl_database = SpectralLineDatabase(config["sl_model"]["fname_db"])
    slm_factory = SpectralLineModelFactory.from_config(
        freq_list, config, sl_db=sl_database
    )
    specie_list = create_specie_list(
        sl_database, obs_data, base_props["spans_include"], config
    )

    with h5py.File(Path(result_dir)/"results_single.h5", "w") as fp:
        if config["opt"]["n_process"] == 1:
            fit_save(
                fp, config, slm_factory, obs_data,
                specie_list, base_props, pool=None
            )
        else:
            use_mpi = config["opt"].get("use_mpi", False)
            with create_pool(config["opt"]["n_process"], use_mpi) as pool:
                fit_save(
                    fp, config, slm_factory, obs_data,
                    specie_list, base_props, pool=pool
                )

    if need_identify:
        identify(config, result_dir, "single")


def fit_save(fp, config, slm_factory, obs_data, specie_list, base_props, pool):
    for item in specie_list:
        print_fitting(item["species"])
        item["id"] = item["id"] + base_props["id_offset"]
        model = FittingModel.from_config(
            slm_factory, [item], obs_data, config,  base_props["T_base"]
        )
        jit_fitting_model(model)
        res_dict = optimize(model, config["opt"], pool)
        grp = fp.create_group(derive_specie_save_name(item))
        save_fitting_result(grp, res_dict)


def create_specie_list(sl_database, obs_data, spans, config):
    if len(spans) == 0:
        peak_mgr = PeakManager(obs_data, **config["peak_manager"])
        freqs = np.mean(np.vstack(peak_mgr.spans_obs_data), axis=1)
    else:
        freqs = np.mean(spans, axis=1)
    return query_species(
        sl_database=sl_database,
        freq_list=get_freq_data(obs_data),
        v_LSR=config["sl_model"].get("vLSR", 0.),
        freqs_include=freqs,
        v_range=config["opt"]["bounds"]["v_LSR"],
        **config["species"],
    )