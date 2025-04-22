from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .optimize import prepare_base_props, optimize, create_pool, print_fitting
from ..config import append_exclude_info
from ..utils import (
    load_result_list, load_result_combine, save_fitting_result,
    derive_specie_save_name
)
from ..preprocess import load_preprocess, get_freq_data
from ..sl_model import (
    combine_specie_lists, create_spectral_line_db,
    SpectralLineModelFactory
)
from ..slm_factory import jit_fitting_model, FittingModel, SpectralLineModelFactory
from ..peaks import (
    derive_peak_params, derive_peaks_multi, derive_intersections
)
from ..identify import identify, Identification


__all__ = ["run_combine"]


def run_combine(config, result_dir, need_identify=True, sl_db=None):
    """Combine all individual fitting results.

    Args:
        config (dict): Config.
        result_dir (str): Directory to save results. This should be the same
            directory to save the results of the individual fitting.
        need_identify (bool):  If ``True``, peform the identification.
    """
    config = deepcopy(config)
    config["opt"]["n_cycle_dim"] = 0

    prominence = config["peak_manager"]["prominence"]
    rel_height = config["peak_manager"]["rel_height"]
    #
    result_dir = Path(result_dir)
    pred_data_list = load_result_list(result_dir/"results_single.h5")
    if len(pred_data_list) == 0:
        return

    pred_data_list.sort(key=lambda item: item["fun"])

    pack_list = []
    for pred_data in pred_data_list:
        pack_list.append(prepare_properties(pred_data, prominence, rel_height))

    fname_base = config.get("fname_base", None)
    if fname_base is not None:
        base_data = load_result_combine(fname_base)
        base_props = prepare_base_props(fname_base, config)
        config_ = append_exclude_info(
            config, base_props["freqs_exclude"], base_props["exclude_list"]
        )
        pack_base = prepare_properties(base_data, prominence, rel_height)
    else:
        config_ = config
        pack_base = None

    with h5py.File(result_dir/"results_combine.h5", "w") as fp:
        combine_greedy(pack_list, pack_base, config_, fp, sl_db=sl_db)

    if need_identify:
        identify(config, result_dir, "combine")


def combine_greedy(pack_list, pack_base, config, fp, sl_db=None):
    config_opt = config["opt"]
    if sl_db is None:
        sl_db = create_spectral_line_db(config["sl_model"]["fname_db"])
    slm_factory = SpectralLineModelFactory(config, sl_db=sl_db)
    obs_info = config["obs_info"]
    idn = Identification(slm_factory, obs_info)
    freq_data = get_freq_data(load_preprocess(obs_info))
    use_f_dice = config["identify"]["use_f_dice"]

    if pack_base is None:
        pack_curr, pack_list, cand_list = derive_first_pack(pack_list, idn, config)
        if pack_curr is None:
            return
        need_opt = True
    else:
        pack_curr = pack_base
        need_opt = False
        cand_list = []

    for pack in pack_list:
        if pack.specie_list is None:
            continue

        if has_intersections(pack_curr.spans, pack.spans) and need_opt:
            res_dict = optimize_with_base(
                pack=pack,
                slm_factory=slm_factory,
                obs_info=obs_info,
                T_base_data=pack_curr.T_pred_data,
                config=config,
                need_init=True,
                need_trail=False
            )
            specie_list_new = pack.specie_list
            params_new = res_dict["x"]
            T_pred_data_new = res_dict["T_pred"]
            spans_new = derive_peaks_multi(
                freq_data, T_pred_data_new,
                idn._peak_mgr.height_list,
                idn._peak_mgr.prom_list,
                idn._peak_mgr.rel_height
            )[0]
            pack_new = Pack(
                specie_list_new, params_new, T_pred_data_new, spans_new
            )

            # Save opt results
            item = pack.specie_list[0]
            save_name = "tmp_{}".format(derive_specie_save_name(item))
            save_fitting_result(fp.create_group(save_name), res_dict)
        else:
            params_new = pack.params
            specie_list_new = pack.specie_list
            pack_new = pack

        # Merge results
        specie_list_combine, params_combine = combine_specie_lists(
            [pack_curr.specie_list, specie_list_new],
            [pack_curr.params, params_new],
        )
        sl_model = slm_factory.create_sl_model(obs_info, specie_list_combine)
        T_pred_data_combine = sl_model(params_combine)

        res = idn.identify(
            specie_list_combine, params_combine, use_f_dice=use_f_dice
        )
        id_new = specie_list_new[0]["id"]
        if check_criteria(res, id_new, config_opt["criteria"]):
            spans_combine = derive_peaks_multi(
                freq_data, T_pred_data_combine,
                idn._peak_mgr.height_list,
                idn._peak_mgr.prom_list,
                idn._peak_mgr.rel_height
            )[0]
            pack_curr = Pack(
                specie_list_combine, params_combine,
                T_pred_data_combine, spans_combine
            )
            need_opt = True
        else:
            cand_list.append(pack_new)

    # Save results
    save_dict = {
        "specie": pack_curr.specie_list,
        "freq": freq_data,
        "T_pred": pack_curr.T_pred_data,
        "x": pack_curr.params,
        "fun": np.nan,
        "nfev": 0,
    }
    save_fitting_result(fp.create_group("combine"), save_dict)

    #
    for pack in cand_list:
        item = pack.specie_list[0]
        save_name = derive_specie_save_name(item)
        if has_intersections(pack_curr.spans, pack.spans):
            res_dict = optimize_with_base(
                pack=pack,
                slm_factory=slm_factory,
                obs_info=obs_info,
                T_base_data=pack_curr.T_pred_data,
                config=config,
                need_init=False,
                need_trail=True
            )
            save_fitting_result(fp.create_group(save_name), res_dict)
        else:
            src_name = "tmp_{}".format(save_name)
            if src_name in fp:
                fp[save_name] = fp[src_name]
            else:
                res_dict = {
                    "specie": pack.specie_list,
                    "freq": freq_data,
                    "T_pred": pack.T_pred_data,
                    "x": pack.params,
                    "fun": np.nan,
                    "nfev": 0,
                }
                save_fitting_result(fp.create_group(save_name), res_dict)


def prepare_properties(pred_data, prominence, rel_height):
    specie_list = pred_data["specie"]
    params = pred_data["x"]
    T_pred_data = pred_data["T_pred"]
    T_back = 0.
    height_list, prom_list \
        = derive_peak_params(prominence, T_back, len(T_pred_data))
    freq_data = pred_data["freq"]
    spans_pred = derive_peaks_multi(
        freq_data=freq_data,
        spec_data=T_pred_data,
        height_list=height_list,
        prom_list=prom_list,
        rel_height=rel_height
    )[0]
    return Pack(specie_list, params, T_pred_data, spans_pred)


def derive_first_pack(pack_list, idn, config):
    cand_list = []
    for i_pack in range(len(pack_list)):
        pack = pack_list[i_pack]
        res = idn.identify(
            pack.specie_list, pack.params,
            use_f_dice=config["identify"]["use_f_dice"]
        )
        key = pack.specie_list[0]["id"]
        if check_criteria(res, key, config["opt"]["criteria"]):
            return pack, pack_list[i_pack+1:], cand_list
        else:
            cand_list.append(pack)
    return None, None, cand_list


def optimize_with_base(pack, slm_factory, obs_info, T_base_data,
                       config, need_init, need_trail):
    print_fitting(pack.specie_list[0]["species"])
    config_opt = deepcopy(config["opt"])
    fitting_model = slm_factory.create_fitting_model(
        obs_info=obs_info,
        specie_list=pack.specie_list,
        T_base_data=T_base_data,
    )
    jit_fitting_model(fitting_model)
    if need_init:
        initial_pos = derive_initial_pos(
            pack.params, fitting_model.bounds,
            config_opt["kwargs_opt"]["nswarm"],
        )
        config_opt["kwargs_opt"]["initial_pos"] = initial_pos
    if not need_trail:
        config_opt["n_trail"] = 1

    res_dict = optimize(fitting_model, config_opt, pool=None)
    return res_dict


def derive_initial_pos(params, bounds, n_swarm):
    assert n_swarm >= 1

    lb, ub = bounds.T
    initial_pos = lb + (ub - lb)*np.random.rand(n_swarm - 1, 1)
    initial_pos = np.vstack([params, initial_pos])
    return initial_pos


def check_criteria(res, id_mol, criteria):
    max_order = get_max_order(criteria)
    if id_mol in res.specie_data:
        score_dict = {"score": res.get_aggregate_prop(id_mol, "score")}
        score_dict.update(res.compute_tx_score(max_order, use_id=True)[id_mol])
    else:
        return False

    for key, cut in criteria.items():
        if score_dict[key] <= cut:
            return False
    return True


def get_max_order(criteria):
    key_list = [key for key in criteria if key.startswith("t")]
    return max(int(key.split("_")[0][1:]) for key in key_list)


def has_intersections(spans_a, spans_b):
    return len(derive_intersections(spans_a, spans_b)[0]) > 0


def get_save_dir(config):
    return Path(config["save_dir"])/Path(config["opt"]["dirname"])


@dataclass
class Pack:
    specie_list: list
    params: object
    T_pred_data: list
    spans: object
