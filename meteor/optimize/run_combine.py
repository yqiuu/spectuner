import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from multiprocessing import Pool

import numpy as np

from .optimize import optimize
from ..preprocess import load_preprocess, get_freq_data
from ..xclass_wrapper import combine_mol_stores
from ..identify import (
    filter_moleclues, derive_peaks_multi, derive_intersections,
    Identification,
)
from ..identify.identify import identify
from ..fitting_model import create_fitting_model


__all__ = ["run_combine"]


def run_combine(config, parent_dir, need_identify=True):
    T_back = config["sl_model"].get("tBack", 0.)
    obs_data = load_preprocess(config["files"], T_back)

    config_opt = config["opt_combine"]
    config_slm = config["sl_model"]
    pool = Pool(config_opt["n_process"])
    prominence = config["pm_loss"]["prominence"]
    rel_height = config["pm_loss"]["rel_height"]

    #
    parent_dir = Path(parent_dir)
    single_dir = parent_dir/"single"
    save_dir = parent_dir/"combine"
    save_dir.mkdir(exist_ok=True)

    pred_data_list = []
    for fname in single_dir.glob("*.pickle"):
        if str(fname.name).startswith("identify"):
            continue
        pred_data_list.append(pickle.load(open(fname, "rb")))
    if len(pred_data_list) == 0:
        raise ValueError("Cannot find any individual fitting results.")
    pred_data_list.sort(key=lambda item: item["cost_best"])

    pack_list = []
    for pred_data in pred_data_list:
        pack_list.append(prepare_properties(
            pred_data, config_slm, T_back, prominence, rel_height, need_filter=True))

    fname_base = config.get("fname_base", None)
    if fname_base is not None:
        base_data = pickle.load(open(fname_base, "rb"))
        pack_base = prepare_properties(
            base_data, config_slm, T_back, prominence, rel_height, need_filter=False)
    else:
        pack_base = None

    combine_greedy(
        pack_list, pack_base, obs_data, config, pool, save_dir,
        force_merge=False
    )

    if need_identify:
        save_name = save_dir/Path("combine.pickle")
        identify(config, parent_dir, save_name)
        identify(config, parent_dir, "combine")


def combine_greedy(pack_list, pack_base, obs_data, config, pool, save_dir, force_merge):
    config_opt = config["opt_combine"]
    config_slm = config["sl_model"]
    T_back = config["sl_model"].get("tBack", 0.)
    prominence = config["pm_loss"]["prominence"]
    rel_height = config["pm_loss"]["rel_height"]
    height = T_back + prominence
    freq_data = get_freq_data(obs_data)
    idn = Identification(obs_data, T_back, prominence, rel_height)

    if pack_base is None:
        pack_curr = pack_list[0]
        pack_list = pack_list[1:]
        need_opt = True
    else:
        pack_curr = pack_base
        need_opt = False
    is_include_list = []

    for pack in pack_list:
        if pack.mol_store is None:
            is_include_list.append(False)
            continue

        spans_inter = derive_intersections(pack_curr.spans, pack.spans)[0]
        if len(spans_inter) > 0 and need_opt:
            res_dict = optimize_with_base(
                pack, obs_data, pack_curr.T_pred_data, config, pool
            )
            params_new =  res_dict["params_best"]
            mol_store_new = pack.mol_store

            # Save opt results
            item = pack.mol_store.mol_list[0]
            save_name = save_dir/Path("tmp_{}_{}.pickle".format(item["id"], item["root"]))
            pickle.dump(res_dict, open(save_name, "wb"))
        else:
            params_new = pack.params
            mol_store_new = pack.mol_store

        # Merge results
        mol_store_combine, params_combine = combine_mol_stores(
            [pack_curr.mol_store, mol_store_new],
            [pack_curr.params, params_new],
            config_slm,
        )
        T_pred_data_combine = mol_store_combine.compute_T_pred_data(
            params_combine, freq_data, config_slm
        )

        res = idn.identify(mol_store_combine, config_slm, params_combine)
        id_new = mol_store_new.mol_list[0]["id"]
        score_new = None
        for id_, score in zip(res.df_mol["id"], res.df_mol["score"]):
            if id_ == id_new:
                score_new = score

        if force_merge or (score_new is not None and score_new > config_opt["score_cut"]):
            spans_combine = derive_peaks_multi(
                freq_data, T_pred_data_combine, height, prominence, rel_height
            )[0]
            pack_curr = Pack(
                mol_store_combine, params_combine,
                T_pred_data_combine, spans_combine
            )
            need_opt = True
            is_include_list.append(True)
        else:
            is_include_list.append(False)

    # Save results
    save_dict = {
        "mol_store": pack_curr.mol_store,
        "freq": get_freq_data(obs_data),
        "T_pred": pack_curr.T_pred_data,
        "params_best": pack_curr.params,
    }
    save_name = save_dir/Path("combine.pickle")
    pickle.dump(save_dict, open(save_name, "wb"))

    # Derive restart index
    idx_restart = 0
    for idx, is_include in enumerate(is_include_list):
        if is_include:
            idx_restart = idx

    #
    for idx, pack in enumerate(pack_list):
        if is_include_list[idx] or pack.mol_store is None:
            continue

        item = pack.mol_store.mol_list[0]
        save_name = save_dir/Path("{}_{}.pickle".format(item["id"], item["root"]))
        if idx < idx_restart:
            res_dict = optimize_with_base(
                pack, obs_data, pack_curr.T_pred_data, config, pool
            )
            pickle.dump(res_dict, open(save_name, "wb"))
        else:
            src_name = save_dir/Path("tmp_{}_{}.pickle".format(item["id"], item["root"]))
            if src_name.exists():
                shutil.copy(src_name, save_name)
            else:
                res_dict = {
                    "mol_store": pack.mol_store,
                    "freq": get_freq_data(obs_data),
                    "T_pred": pack.T_pred_data,
                    "params_best": pack.params,
                }
                pickle.dump(res_dict, open(save_name, "wb"))

    return pack_curr


def prepare_properties(pred_data, config_slm, T_back, prominence, rel_height, need_filter):
    height = T_back + prominence
    mol_store = pred_data["mol_store"]
    params = pred_data["params_best"]
    T_pred_data = pred_data["T_pred"]
    freq_data = pred_data["freq"]
    spans_pred = derive_peaks_multi(
        freq_data=freq_data,
        spec_data=T_pred_data,
        height=height,
        prominence=prominence,
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
            T_back=T_back,
            prominence=prominence,
            rel_height=rel_height
        )
    else:
        mol_store_new = mol_store
        params_new = params
    return Pack(mol_store_new, params_new, T_pred_data, spans_pred)


def optimize_with_base(pack, obs_data, T_base, config, pool):
    config_opt = config["opt_combine"]
    model = create_fitting_model(
        obs_data, pack.mol_store, config, config_opt, T_base
    )
    initial_pos = derive_initial_pos(
        pack.params, model.bounds, config_opt["kwargs_opt"]["nswarm"],
    )
    config_opt["kwargs_opt"]["initial_pos"] = initial_pos
    res_dict = optimize(model, config_opt, pool)
    return res_dict


def derive_initial_pos(params, bounds, n_swarm):
    assert n_swarm >= 1

    lb, ub = bounds.T
    initial_pos = lb + (ub - lb)*np.random.rand(n_swarm - 1, 1)
    initial_pos = np.vstack([params, initial_pos])
    return initial_pos


def get_save_dir(config):
    return Path(config["save_dir"])/Path(config["opt_combine"]["dirname"])


@dataclass
class Pack:
    mol_store: object
    params: object
    T_pred_data: list
    spans: object
