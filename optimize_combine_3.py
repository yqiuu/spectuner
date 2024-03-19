import sys
import yaml
import pickle
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
    height = T_back + prominence
    #
    idn = Identification(obs_data, T_back, prominence, rel_height)

    # Read pred data
    dir_single = Path(config["save_dir"])/Path(config["opt_single"]["dirname"])
    pred_data_list = []
    for fname in dir_single.glob("*.pickle"):
        pred_data_list.append(pickle.load(open(fname, "rb")))
    pred_data_list.sort(key=lambda item: item["cost_best"])

    i_group = 0
    save_dir = Path(config["save_dir"])/Path(config_opt["dirname"])
    freq_data = pred_data_list[0]["freq"]
    mol_store_curr = None
    for pred_data in pred_data_list:
        mol_store_new, params_new, T_pred_data_new, spans_new = prepare_properties(
            pred_data, config_slm, T_back, prominence, rel_height
        )
        if mol_store_new is None:
            continue

        if mol_store_curr is None:
            mol_store_combine = mol_store_new
            params_combine = params_new
            T_pred_data_combine = T_pred_data_new
            score_new = config_opt["score_cut"] + 1 # Force save results
        else:
            spans_inter = derive_intersections(spans_curr, spans_new)[0]
            if len(spans_inter) > 0:
                model = create_model(
                    obs_data, mol_store_new, config, T_pred_data_curr
                )
                initial_pos = derive_initial_pos(
                    params_new, model.bounds, config_opt["kwargs_opt"]["nswarm"],
                )
                config_opt["kwargs_opt"]["initial_pos"] = initial_pos
                res_dict = optimize(model, config_opt, pool)
                params_new =  res_dict["params_best"]
            else:
                res_dict = {}

            mol_store_combine, params_combine = combine_mol_stores(
                [mol_store_curr, mol_store_new],
                [params_curr, params_new],
                config_slm,
            )
            T_pred_data_combine = mol_store_combine.compute_T_pred_data(
                params_combine, freq_data, config_slm
            )
            res = idn.identify(
                mol_store_combine, config_slm, params_combine, T_pred_data_combine
            )
            id_new = mol_store_new.mol_list[0]["id"]
            score_new = None
            for id_, score in zip(res.df_mol["id"], res.df_mol["score"]):
                if id_ == id_new:
                    score_new = score

        if score_new is not None and score_new > config_opt["score_cut"]:
            mol_store_curr = mol_store_combine
            params_curr = params_combine
            T_pred_data_curr = T_pred_data_combine
            spans_curr = derive_peaks_multi(
                freq_data, T_pred_data_combine, height, prominence, rel_height
            )[0]

            save_data = {
                "mol_store": mol_store_curr,
                "freq": freq_data,
                "T_pred": T_pred_data_combine,
                "params_best": params_curr,
                "mol_store_new": mol_store_new,
            }
            save_name = Path("group_{}.pickle".format(i_group))
            i_group += 1
        else:
            save_data = res_dict
            save_data["T_pred_combine"] = T_pred_data_combine
            if score_new is not None:
                save_data["idn_res"] = res.extract_sub(id_new)
            save_name = Path("sub_res_id{}.pickle".format(id_new))
        pickle.dump(save_data, open(save_dir/Path(save_name), "wb"))


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
    return mol_store_new, params_new, T_pred_data, spans_pred


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


def combine_all(save_dir, config_slm):
    mol_store_list = []
    params_list = []
    for fname in save_dir.glob("group_*_final.pickle"):
        data = pickle.load(open(fname, "rb"))
        mol_store_list.append(data["mol_store"])
        params_list.append(data["params_best"])
    mol_store_new, params_new = combine_mol_stores(mol_store_list, params_list, config_slm)
    sl_model = mol_store_new.create_spectral_line_model(config_slm)
    iterator = sl_model.call_multi(
        data["freq"], mol_store_new.include_list, params_new, remove_dir=True
    )
    T_pred_data = [args[0] for args in iterator]
    res_dict = {
        "mol_store": mol_store_new,
        "freq": data["freq"],
        "T_pred": T_pred_data,
        "params_best": params_new
    }
    pickle.dump(res_dict, open(save_dir/Path("combine.pickle"), "wb"))


def derive_initial_pos(params, bounds, n_swarm):
    assert n_swarm >= 1

    lb, ub = bounds.T
    initial_pos = lb + (ub - lb)*np.random.rand(n_swarm - 1, 1)
    initial_pos = np.vstack([params, initial_pos])
    return initial_pos


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    main(config)