import pickle
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import append_exclude_info
from .optimize import prepare_base_props, optimize, create_pool
from ..utils import load_pred_data
from ..preprocess import load_preprocess, get_freq_data
from ..xclass_wrapper import combine_mol_stores
from ..identify import (
    filter_moleclues, derive_peak_params,
    derive_peaks_multi, derive_intersections,
    PeakManager,
)
from ..identify.identify import identify
from ..fitting_model import create_fitting_model


__all__ = ["run_combine"]


def run_combine(config, parent_dir, need_identify=True):
    use_mpi = config["opt"].get("use_mpi", False)
    with create_pool(config["opt"]["n_process"], use_mpi) as pool:
        config = deepcopy(config)
        config["opt"]["n_cycle_dim"] = 0

        T_back = config["sl_model"].get("tBack", 0.)
        obs_data = load_preprocess(config["files"], T_back)
        config_slm = config["sl_model"]

        prominence = config["peak_manager"]["prominence"]
        rel_height = config["peak_manager"]["rel_height"]
        #
        parent_dir = Path(parent_dir)
        single_dir = parent_dir/"single"
        save_dir = parent_dir/"combine"
        save_dir.mkdir(exist_ok=True)

        pred_data_list = load_pred_data(single_dir.glob("*.pickle"), reset_id=False)
        if len(pred_data_list) == 0:
            raise ValueError("Cannot find any individual fitting results.")
        pred_data_list.sort(key=lambda item: item["cost_best"])

        pack_list = []
        for pred_data in pred_data_list:
            pack_list.append(prepare_properties(
                pred_data, config_slm, T_back, prominence, rel_height, need_filter=False))

        fname_base = config.get("fname_base", None)
        if fname_base is not None:
            base_data = pickle.load(open(fname_base, "rb"))
            base_props = prepare_base_props(fname_base, config)
            config_ = append_exclude_info(
                config, base_props["freqs_exclude"], base_props["exclude_list"]
            )
            pack_base = prepare_properties(
                base_data, config_slm, T_back, prominence, rel_height, need_filter=False)
        else:
            config_ = config
            pack_base = None

        combine_greedy(
            pack_list, pack_base, obs_data, config_, pool, save_dir
        )

    if need_identify:
        save_name = save_dir/Path("combine.pickle")
        if save_name.exists():
            identify(config, parent_dir, save_name)
            identify(config, parent_dir, "combine")


def combine_greedy(pack_list, pack_base, obs_data, config, pool, save_dir):
    config_opt = config["opt"]
    T_back = config["sl_model"].get("tBack", 0.)
    freq_data = get_freq_data(obs_data)
    peak_mgr = PeakManager(obs_data, T_back, **config["peak_manager"])

    if pack_base is None:
        pack_curr, pack_list, cand_list = derive_first_pack(pack_list, peak_mgr, config)
        if pack_curr is None:
            return
        need_opt = True
    else:
        pack_curr = pack_base
        need_opt = False
        cand_list = []

    for pack in pack_list:
        if pack.mol_store is None:
            continue

        if has_intersections(pack_curr.spans, pack.spans) and need_opt:
            res_dict = optimize_with_base(
                pack, obs_data, pack_curr.T_pred_data, config, pool,
                need_init=True, need_trail=False
            )
            mol_store_new = pack.mol_store
            params_new = res_dict["params_best"]
            T_pred_data_new = res_dict["T_pred"]
            spans_new = derive_peaks_multi(
                freq_data, T_pred_data_new,
                peak_mgr.height_list, peak_mgr.prom_list, peak_mgr.rel_height
            )[0]
            pack_new = Pack(
                mol_store_new, params_new, T_pred_data_new, spans_new
            )

            # Save opt results
            item = pack.mol_store.mol_list[0]
            save_name = save_dir/Path("tmp_{}_{}.pickle".format(item["root"], item["id"]))
            pickle.dump(res_dict, open(save_name, "wb"))
        else:
            params_new = pack.params
            mol_store_new = pack.mol_store
            pack_new = pack

        # Merge results
        mol_store_combine, params_combine = combine_mol_stores(
            [pack_curr.mol_store, mol_store_new],
            [pack_curr.params, params_new],
        )
        T_pred_data_combine = mol_store_combine.compute_T_pred_data(
            params_combine, freq_data, config
        )

        res = peak_mgr.identify(mol_store_combine, config, params_combine)
        id_new = mol_store_new.mol_list[0]["id"]
        if check_criteria(res, id_new, config_opt["criteria"]):
            spans_combine = derive_peaks_multi(
                freq_data, T_pred_data_combine,
                peak_mgr.height_list, peak_mgr.prom_list, peak_mgr.rel_height
            )[0]
            pack_curr = Pack(
                mol_store_combine, params_combine,
                T_pred_data_combine, spans_combine
            )
            need_opt = True
        else:
            cand_list.append(pack_new)

    # Save results
    save_dict = {
        "mol_store": pack_curr.mol_store,
        "freq": freq_data,
        "T_pred": pack_curr.T_pred_data,
        "params_best": pack_curr.params,
    }
    save_name = save_dir/Path("combine.pickle")
    pickle.dump(save_dict, open(save_name, "wb"))

    #
    for pack in cand_list:
        item = pack.mol_store.mol_list[0]
        save_name = save_dir/Path("{}_{}.pickle".format(item["root"], item["id"]))
        if has_intersections(pack_curr.spans, pack.spans):
            res_dict = optimize_with_base(
                pack, obs_data, pack_curr.T_pred_data, config, pool,
                need_init=False, need_trail=True
            )
            pickle.dump(res_dict, open(save_name, "wb"))
        else:
            src_name = save_dir/Path("tmp_{}_{}.pickle".format(item["root"], item["id"]))
            if src_name.exists():
                shutil.copy(src_name, save_name)
            else:
                res_dict = {
                    "mol_store": pack.mol_store,
                    "freq": freq_data,
                    "T_pred": pack.T_pred_data,
                    "params_best": pack.params,
                }
                pickle.dump(res_dict, open(save_name, "wb"))


def prepare_properties(pred_data, config_slm, T_back_data,
                       prominence, rel_height, need_filter):
    mol_store = pred_data["mol_store"]
    params = pred_data["params_best"]
    T_pred_data = pred_data["T_pred"]
    height_list, prom_list \
        = derive_peak_params(prominence, T_back_data, len(T_pred_data))
    freq_data = pred_data["freq"]
    spans_pred = derive_peaks_multi(
        freq_data=freq_data,
        spec_data=T_pred_data,
        height_list=height_list,
        prom_list=prom_list,
        rel_height=rel_height
    )[0]
    # Filter molecules
    if need_filter:
        mol_store_new, params_new = filter_moleclues(
            mol_store=mol_store,
            config_slm=config_slm,
            params=params,
            freq_data=freq_data,
            T_pred_data=T_pred_data,
            T_back=T_back_data,
            prominence=prominence,
            rel_height=rel_height
        )
    else:
        mol_store_new = mol_store
        params_new = params
    return Pack(mol_store_new, params_new, T_pred_data, spans_pred)


def derive_first_pack(pack_list, peak_mgr, config):
    cand_list = []
    for i_pack in range(len(pack_list)):
        pack = pack_list[i_pack]
        res = peak_mgr.identify(pack.mol_store, config, pack.params)
        key = pack.mol_store.mol_list[0]["id"]
        if check_criteria(res, key, config["opt"]["criteria"]):
            return pack, pack_list[i_pack+1:], cand_list
        else:
            cand_list.append(pack)
    return None, None, cand_list


def optimize_with_base(pack, obs_data, T_base_data,
                       config, pool, need_init, need_trail):
    config_opt = deepcopy(config["opt"])
    model = create_fitting_model(
        obs_data, pack.mol_store, config, T_base_data
    )
    if need_init:
        initial_pos = derive_initial_pos(
            pack.params, model.bounds, config_opt["kwargs_opt"]["nswarm"],
        )
        config_opt["kwargs_opt"]["initial_pos"] = initial_pos
    if not need_trail:
        config_opt["n_trail"] = 1
    res_dict = optimize(model, config_opt, pool)
    return res_dict


def derive_initial_pos(params, bounds, n_swarm):
    assert n_swarm >= 1

    lb, ub = bounds.T
    initial_pos = lb + (ub - lb)*np.random.rand(n_swarm - 1, 1)
    initial_pos = np.vstack([params, initial_pos])
    return initial_pos


def check_criteria(res, id_mol, criteria):
    max_order = get_max_order(criteria)
    if id_mol in res.mol_data:
        score_dict = {"score": res.get_aggregate_prop(id_mol, "score")}
        score_dict.update(res.compute_tx_score(max_order)[id_mol])
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
    mol_store: object
    params: object
    T_pred_data: list
    spans: object
