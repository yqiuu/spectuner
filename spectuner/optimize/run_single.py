from pathlib import Path

import h5py
import numpy as np

from .optimize import prepare_base_props, optimize_all
from .. import ai
from ..config import append_exclude_info
from ..sl_model import query_species, select_master_name, create_spectral_line_db
from ..slm_factory import SpectralLineModelFactory
from ..peaks import PeakManager
from ..identify import identify
from ..utils import (
    save_fitting_result, derive_specie_save_name, create_process_pool
)


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
    with create_process_pool(config["n_process"]) as pool:
        if config["inference"]["ckpt"] is None:
            engine = slm_factory
        else:
            engine = ai.InferenceModel.from_config(config, sl_db=sl_db)
        results = optimize_all(
            engine=engine,
            obs_info=config["obs_info"],
            targets=targets,
            config_opt=config["opt_single"],
            T_base_data=base_props["T_base"],
            trans_counts=trans_counts,
            config_inf=config["inference"],
            pool=pool,
        )

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(result_dir/"results_single.h5", "w") as fp:
        for res in results:
            grp = fp.create_group(derive_specie_save_name(res["specie"][0]))
            save_fitting_result(grp, res)

    if need_identify:
        identify(config, result_dir, "single", sl_db=sl_db)


def create_specie_list(sl_db, id_offset, spans, config):
    if len(spans) == 0:
        peak_mgr = PeakManager.from_config(config)
        freqs = peak_mgr.freqs_peak
        freq_data = peak_mgr.freq_data
    else:
        freqs = np.mean(spans, axis=1)
        freq_data = [item["spec"][:, 0] for item in config["obs_info"]]

    groups, trans_counts = query_species(
        sl_db=sl_db,
        freq_data=freq_data,
        v_LSR=config["sl_model"].get("vLSR", 0.),
        freqs_include=freqs,
        v_range=config["param_info"]["v_offset"]["bound"],
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