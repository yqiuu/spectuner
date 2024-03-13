import sys
import yaml
import pickle
from pathlib import Path
from multiprocessing import Pool

import numpy as np

from src.preprocess import load_preprocess
from src.fitting_model import FittingModel
from src.algorithms import filter_moleclues, derive_blending_list
from src.optimize import (
    optimize, combine_mol_stores, random_mutation_by_group,
)


def main(config):
    T_back = config["sl_model"].get("tBack", 0.)
    obs_data = load_preprocess(config["file_spec"], T_back)

    config_opt = config["opt_combine"]
    pool = Pool(config_opt["n_process"])
    prominence = config_opt["pm_loss"]["prominence"]
    rel_height = config_opt["pm_loss"]["rel_height"]

    # Read pred data
    dir_single = Path(config["save_dir"])/Path(config["opt_single"]["dirname"])
    pred_data_list = []
    for fname in dir_single.glob("*.pickle"):
        pred_data_list.append(pickle.load(open(fname, "rb")))

    blending_list = derive_blending_list(
        obs_data, pred_data_list, T_back, prominence, rel_height)
    #
    freq_data = [spec[:, 0] for spec in obs_data]

    #
    save_dir = Path(config["save_dir"])/Path(config_opt["dirname"])
    i_group = 0
    i_save = 0
    mol_store_curr = None
    while len(blending_list) != 0:
        if mol_store_curr is None:
            i_pred, inds_peak_curr = blending_list.pop(0)
            pred_data = pred_data_list[i_pred]
            mol_store_curr = pred_data["mol_store"]
            params_curr = pred_data["params_best"]
            T_pred_data_curr = pred_data["T_pred"]

        # Filter molecules
        mol_store_curr, params_curr = filter_moleclues(
            mol_store=mol_store_curr,
            config_slm=config["sl_model"],
            params=params_curr,
            freq_data=freq_data,
            T_pred_data=T_pred_data_curr,
            T_back=T_back,
            prominence=prominence,
            rel_height=rel_height
        )

        # Find blending
        idx_pop = None
        for idx_blen, (i_pred, inds_peak) in enumerate(blending_list):
            if not inds_peak_curr.isdisjoint(inds_peak):
                pred_data = pred_data_list[i_pred]
                mol_store_blen, params_blen = filter_moleclues(
                    mol_store=pred_data["mol_store"],
                    config_slm=config["sl_model"],
                    params=pred_data["params_best"],
                    freq_data=pred_data["freq"],
                    T_pred_data=pred_data["T_pred"],
                    T_back=T_back,
                    prominence=prominence,
                    rel_height=rel_height
                )
                mol_store_curr, params_curr = combine_mol_stores(
                    [mol_store_curr, mol_store_blen],
                    [params_curr, params_blen],
                    config["sl_model"]
                )
                inds_peak_curr.update(inds_peak)
                model = create_model(obs_data, mol_store_curr, config)
                initial_pos = derive_initial_pos(
                    model.sl_model.pm, params_curr, model.bounds,
                    n_swarm=config_opt["kwargs_opt"]["nswarm"],
                    prob=config_opt["prob_mutation"]
                )
                config_opt["kwargs_opt"]["initial_pos"] = initial_pos
                res_dict = optimize(model, config_opt, pool)
                params_curr = res_dict["params_best"]
                T_pred_data_curr = res_dict["T_pred"]
                save_name = Path("group_{}_{}.pickle".format(
                    i_group, i_save))
                i_save += 1
                pickle.dump(res_dict, open(save_dir/save_name, "wb"))
                idx_pop = idx_blen
                break
        else:
            sl_model = mol_store_curr.create_spectral_line_model(config["sl_model"])
            iterator = sl_model.call_multi(
                freq_data, mol_store_curr.include_list, params_curr, remove_dir=True
            )
            T_pred_data = [args[0] for args in iterator]
            res_dict = {
                "mol_store": mol_store_curr,
                "freq": freq_data,
                "T_pred": T_pred_data,
                "params_best": params_curr
            }
            save_name = Path("group_{}_final.pickle".format(
                i_group, len(mol_store_curr.mol_list)))
            pickle.dump(res_dict, open(save_dir/save_name, "wb"))
            i_group += 1
            i_save = 0
            mol_store_curr = None

        if idx_pop is not None:
            blending_list.pop(idx_pop)

    combine_all(save_dir, config["sl_model"])


def create_model(obs_data, mol_store, config):
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


def derive_initial_pos(pm, params, bounds, n_swarm, prob):
    assert n_swarm >= 2

    initial_pos = [params]
    initial_pos.append(random_mutation_by_group(pm, params, bounds, prob=1.))
    for _ in range(n_swarm - 2):
        initial_pos.append(random_mutation_by_group(pm, params, bounds, prob))
    initial_pos = np.vstack(initial_pos)
    return initial_pos


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    main(config)