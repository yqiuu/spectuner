import sys
import yaml
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from multiprocessing import Pool

import numpy as np

from src.preprocess import load_preprocess
from src.fitting_model import FittingModel
from src.algorithms import (
    filter_moleclues, derive_peaks_multi, derive_intersections,
    Identification
)
from src.optimize import (
    optimize, combine_mol_stores,
)


def main(config):
    T_back = config["sl_model"].get("tBack", 0.)
    obs_data = load_preprocess(config["file_spec"], T_back)

    config_opt = config["opt_combine"]
    config_slm = config["sl_model"]
    pool = Pool(config_opt["n_process"])
    prominence = config_opt["pm_loss"]["prominence"]
    rel_height = config_opt["pm_loss"]["rel_height"]

    dir_single = Path(config["save_dir"])/Path(config["opt_single"]["dirname"])
    pred_data_list = []
    for fname in dir_single.glob("*.pickle"):
        pred_data_list.append(pickle.load(open(fname, "rb")))
    pred_data_list.sort(key=lambda item: item["cost_best"])
    order_list = [pred_data["mol_store"].mol_list[0]["id"] for pred_data in pred_data_list]

    pack_list = []
    for pred_data in pred_data_list:
        pack_list.append(prepare_properties(
            pred_data, config_slm, T_back, prominence, rel_height))

    pack_final = combine_greedy(
        pack_list[0], pack_list[1:], obs_data, config, pool,
        force_merge=False
    )

    # Save results
    save_dir = Path(config["save_dir"])/Path(config["opt_combine"]["dirname"])
    save_dict = {
        "mol_store": pack_final.mol_store,
        "freq": get_freq_data(obs_data),
        "T_pred": pack_final.T_pred_data,
        "params_best": pack_final.params,
        "order_list": order_list
    }
    save_name = save_dir/Path("combine.pickle")
    pickle.dump(save_dict, open(save_name, "wb"))

    #
    include_list = [item["id"] for item in pack_final.mol_store.mol_list]
    exclude_list = [i_id for i_id in order_list if i_id not in include_list]
    idx_restart = 0
    for i_id in include_list:
        idx_restart = max(idx_restart, order_list.index(i_id))

    for i_id in exclude_list:
        idx = order_list.index(i_id)
        pack = pack_list[idx]
        item = pack.mol_store.mol_list[0]
        save_name = save_dir/Path("{}_{}.pickle".format(item["id"], item["root"]))
        if idx > idx_restart:
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
        else:
            res_dict = optimize_with_base(
                pack, obs_data, pack_final.T_pred_data, config_opt, pool
            )
            pickle.dump(res_dict, open(save_name, "wb"))


def combine_greedy(pack_0, pack_list, obs_data, config, pool, force_merge):
    config_opt = config["opt_combine"]
    config_slm = config["sl_model"]
    save_dir = Path(config["save_dir"])/Path(config_opt["dirname"])
    T_back = config["sl_model"].get("tBack", 0.)
    prominence = config_opt["pm_loss"]["prominence"]
    rel_height = config_opt["pm_loss"]["rel_height"]
    height = T_back + prominence
    freq_data = get_freq_data(obs_data)
    idn = Identification(obs_data, T_back, prominence, rel_height)

    pack_curr = pack_0
    for pack in pack_list:
        if pack.mol_store is None:
            continue

        spans_inter = derive_intersections(pack_curr.spans, pack.spans)[0]
        if len(spans_inter) > 0:
            res_dict = optimize_with_base(
                pack, obs_data, pack_curr.T_pred_data, config_opt, pool
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

    return pack_curr


def prepare_properties(pred_data, config_slm, T_back, prominence, rel_height):
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
    return Pack(mol_store_new, params_new, T_pred_data, spans_pred)


def optimize_with_base(pack, obs_data, T_base, config_opt, pool):
    model = create_model(
        obs_data, pack.mol_store, config, T_base
    )
    initial_pos = derive_initial_pos(
        pack.params, model.bounds, config_opt["kwargs_opt"]["nswarm"],
    )
    config_opt["kwargs_opt"]["initial_pos"] = initial_pos
    res_dict = optimize(model, config_opt, pool)
    return res_dict

def create_model(obs_data, mol_store, config, base_data):
    pm = mol_store.create_parameter_manager(config["sl_model"])
    # TODO: better way to create bounds?
    config_opt = config["opt_combine"]
    bounds = pm.scaler.derive_bounds(
        pm,
        config_opt["bounds_mol"],
        config_opt["bounds_iso"],
        config_opt["bounds_misc"]
    )
    model = FittingModel(
        obs_data, mol_store, bounds, config["sl_model"],
        config_pm_loss=config_opt.get("pm_loss", None),
        config_thr_loss=config_opt.get("thr_loss", None),
        base_data=base_data,
    )
    return model


def derive_initial_pos(params, bounds, n_swarm):
    assert n_swarm >= 1

    lb, ub = bounds.T
    initial_pos = lb + (ub - lb)*np.random.rand(n_swarm - 1, 1)
    initial_pos = np.vstack([params, initial_pos])
    return initial_pos


def get_freq_data(obs_data):
    return [spec[:, 0] for spec in obs_data]


@dataclass
class Pack:
    mol_store: object
    params: object
    T_pred_data: list
    spans: object


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    main(config)