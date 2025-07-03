from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, Counter

import h5py
import numpy as np
from tqdm import tqdm

from .optimize import (
    prepare_base_props, optimize, optimize_all, print_fitting, join_specie_names
)
from .. import ai
from ..config import append_exclude_info
from ..utils import (
    load_result_list, load_result_combine, save_fitting_result,
    derive_specie_save_name, create_process_pool
)
from ..preprocess import load_preprocess, get_freq_data
from ..sl_model import create_spectral_line_db
from ..slm_factory import (
    jit_fitting_model, combine_specie_lists, SpectralLineModelFactory
)
from ..peaks import (
    derive_peak_params, derive_peaks_multi,
    derive_intersections, derive_prominence
)
from ..identify import identify, Identification


__all__ = ["run_combining_line_id"]


def run_combining_line_id(config, result_dir, need_identify=True, sl_db=None):
    """Combine all individual fitting results.

    Args:
        config (dict): Config.
        result_dir (str): Directory to save results. This should be the same
            directory to save the results of the individual fitting.
        need_identify (bool):  If ``True``, peform the identification.
    """
    prominence = derive_prominence(config)
    rel_height = config["peak_manager"]["rel_height"]
    #
    result_dir = Path(result_dir)
    pred_data_list = load_result_list(result_dir/"results_single.h5")
    if len(pred_data_list) == 0:
        return

    pred_data_list = sort_result_list(
        pred_data_list, version_mode=config["species"]["version_mode"]
    )

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

    result = combine_greedy(
        pack_list, pack_base, config_, sl_db=sl_db
    )
    if result is not None:
        with h5py.File(result_dir/"results_combine.h5", "w") as fp:
            combine_dict, save_dict = result
            save_fitting_result(fp.create_group("combine"), combine_dict)
            for save_name, res in save_dict.items():
                save_fitting_result(fp.create_group(save_name), res)

        if need_identify:
            identify(config, result_dir, "combine")


def combine_greedy(pack_list, pack_base, config, sl_db=None):
    if sl_db is None:
        sl_db = create_spectral_line_db(config["sl_model"]["fname_db"])
    slm_factory = SpectralLineModelFactory(config, sl_db=sl_db)
    if config["inference"]["ckpt"] is None:
        engine = slm_factory
    else:
        engine = ai.InferenceModel.from_config(config, sl_db=sl_db)
    obs_info = config["obs_info"]
    idn = Identification(slm_factory, obs_info)
    freq_data = get_freq_data(load_preprocess(obs_info))
    use_f_dice = config["identify"]["use_f_dice"]
    criteria = config["identify"]["criteria"]

    if pack_base is None:
        pack_curr, pack_list, cand_list = derive_first_pack(pack_list, idn, config)
        if pack_curr is None:
            return
        need_opt = True
    else:
        pack_curr = pack_base
        need_opt = False
        cand_list = []

    save_dict = {}
    pbar = tqdm(pack_list)
    for pack in pbar:
        if pack.specie_list is None:
            continue

        if has_intersections(pack_curr.spans, pack.spans) and need_opt:
            pbar.set_description("Combining {}".format(
                join_specie_names(pack.specie_list[0]["species"])))
            res_dict = optimize(
                engine=engine,
                obs_info=obs_info,
                specie_list=pack.specie_list,
                config=config,
                T_base_data=pack_curr.T_pred_data,
                x0=pack.params,
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
            save_dict[save_name] = res_dict
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
        if check_criteria(res, id_new, criteria):
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

    pbar.close()

    # Set result
    combine_dict = {
        "specie": pack_curr.specie_list,
        "freq": freq_data,
        "T_pred": pack_curr.T_pred_data,
        "x": pack_curr.params,
        "fun": np.nan,
        "nfev": 0,
    }

    # Process candidates
    targets = []
    target_names = []
    for pack in cand_list:
        item = pack.specie_list[0]
        save_name = derive_specie_save_name(item)
        if has_intersections(pack_curr.spans, pack.spans):
            targets.append(pack.specie_list)
            target_names.append(save_name)
        else:
            src_name = "tmp_{}".format(save_name)
            if src_name in save_dict:
                save_dict[save_name] = save_dict[src_name]
            else:
                save_dict[save_name] = {
                    "specie": pack.specie_list,
                    "freq": freq_data,
                    "T_pred": pack.T_pred_data,
                    "x": pack.params,
                    "fun": np.nan,
                    "nfev": 0,
                }

    if isinstance(engine, ai.InferenceModel):
        entries = sl_db.query_transitions(freq_data)
        trans_counts = Counter([name for name, _ in entries])
    else:
        trans_counts = None

    with create_process_pool(config["n_process"]) as pool:
        results = optimize_all(
            engine=engine,
            obs_info=obs_info,
            targets=targets,
            config=config,
            T_base_data=pack_curr.T_pred_data,
            trans_counts=trans_counts,
            pool=pool,
        )
    for save_name, res in zip(target_names, results):
        save_dict[save_name] = res

    return combine_dict, save_dict


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
        if check_criteria(res, key, config["identify"]["criteria"]):
            return pack, pack_list[i_pack+1:], cand_list
        else:
            cand_list.append(pack)
    return None, None, cand_list


def optimize_with_base(pack, slm_factory, obs_info, T_base_data,
                       config, need_init, need_trial):
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
    if not need_trial:
        config_opt["n_trial"] = 1

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


def sort_result_list(pred_data_list, version_mode):
    if version_mode == "all":
        tmp_dict = defaultdict(list)
        for pred_data in pred_data_list:
            key = pred_data["specie"][0]["root"]
            key = ";".join(key.split(";")[:-1])
            tmp_dict[key].append(pred_data)
        for res_list in tmp_dict.values():
            res_list.sort(key=lambda item: item["fun"])
        pred_data_list_ = []
        for res_list in tmp_dict.values():
            pred_data_list_.append(res_list[0])
    else:
        pred_data_list_ = pred_data_list

    pred_data_list_.sort(key=lambda item: item["fun"])
    return pred_data_list_


def has_intersections(spans_a, spans_b):
    return len(derive_intersections(spans_a, spans_b)[0]) > 0


@dataclass
class Pack:
    specie_list: list
    params: object
    T_pred_data: list
    spans: object
