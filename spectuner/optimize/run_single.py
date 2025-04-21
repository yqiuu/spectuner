import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np

from .optimize import prepare_base_props, optimize, create_optimizer, print_fitting
from ..config import append_exclude_info
from ..preprocess import load_preprocess, get_freq_data
from ..sl_model import query_species, select_master_name, create_spectral_line_db
from ..slm_factory import jit_fitting_model, SpectralLineModelFactory
from ..ai import predict_single_pixel, InferenceModel
from ..peaks import PeakManager
from ..identify import identify
from ..utils import save_fitting_result, derive_specie_save_name


__all__ = ["run_single", "create_specie_list"]


def run_single(config, result_dir, need_identify=True, sl_db=None):
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
    if sl_db is None:
        sl_db = create_spectral_line_db(config["sl_model"]["fname_db"])
    slm_factory = SpectralLineModelFactory(config, sl_db=sl_db)

    targets, trans_counts = create_specie_list(
        sl_db, base_props["id_offset"],
        base_props["spans_include"], config
    )
    with mp.Pool(config["opt"]["n_process"]) as pool:
        if config["inference"]["ckpt"] is None:
            results = fit_all(
                slm_factory=slm_factory,
                obs_info=config["obs_info"],
                targets=targets,
                base_props=base_props,
                config_opt=config["opt"],
            )
        else:
            inf_model = InferenceModel.from_config(config, sl_db=sl_db)
            results = fit_all_with_agent(
                inf_model=inf_model,
                obs_info=config["obs_info"],
                targets=targets,
                trans_counts=trans_counts,
                config_opt=config["opt"],
                config_inf=config["inference"],
                pool=pool
            )

    with h5py.File(Path(result_dir)/"results_single.h5", "w") as fp:
        for res in results:
            grp = fp.create_group(derive_specie_save_name(res["specie"][0]))
            save_fitting_result(grp, res)

    if need_identify:
        identify(config, result_dir, "single", sl_db=sl_db)


def fit_all(slm_factory: SpectralLineModelFactory,
            obs_info: list,
            targets: list,
            base_props: dict,
            config_opt: dict) -> list:
    results = []
    for specie_list in targets:
        print_fitting(specie_list[0]["species"])
        fitting_model = slm_factory.create_fitting_model(
            obs_info=obs_info,
            specie_list=specie_list,
            T_base_data=base_props["T_base"],
        )
        jit_fitting_model(fitting_model)
        res_dict = optimize(fitting_model, config_opt, pool=None)
        results.append(res_dict)
    return results


def fit_all_with_agent(inf_model: InferenceModel,
                       obs_info: list,
                       targets: list,
                       trans_counts: dict,
                       config_opt: dict,
                       config_inf: dict,
                       pool: mp.Pool) -> list:
    specie_names = []
    numbers = []
    for specie_list in targets:
        name = specie_list[0]["root"]
        specie_names.append(name)
        numbers.append(trans_counts[name])
    opt = create_optimizer(config_opt)
    results = predict_single_pixel(
        inf_model=inf_model,
        obs_info=obs_info,
        entries=specie_names,
        numbers=numbers,
        postprocess=opt,
        max_diff=config_inf["max_diff"],
        max_batch_size=config_inf["max_batch_size"],
        device=config_inf["device"],
        pool=pool
    )
    for res, specie_list in zip(results, targets):
        res["specie"] = specie_list
    return results


def create_specie_list(sl_db, id_offset, spans, config):
    obs_data = load_preprocess(config["obs_info"])
    if len(spans) == 0:
        peak_mgr = PeakManager(obs_data, **config["peak_manager"])
        freqs = np.mean(np.vstack(peak_mgr.spans_obs_data), axis=1)
    else:
        freqs = np.mean(spans, axis=1)

    groups, trans_counts = query_species(
        sl_db=sl_db,
        freq_data=get_freq_data(obs_data),
        v_LSR=config["sl_model"].get("vLSR", 0.),
        freqs_include=freqs,
        v_range=config["opt"]["bounds"]["v_LSR"],
        **config["species"],
    )

    idx = id_offset
    targets = []
    for species in groups:
        root_name = select_master_name(species)
        if root_name is None:
            continue
        targets.append([{"id": idx, "root": root_name, "species": species}])
        idx += 1
    return targets, trans_counts