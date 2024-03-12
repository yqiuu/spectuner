import sys
import yaml
import pickle
from pathlib import Path
from multiprocessing import Pool

import numpy as np

from src.preprocess import load_preprocess
from src.fitting_model import create_fitting_model
from src.xclass_wrapper import create_wrapper_from_config
from src.algorithms import filter_moleclues, derive_blending_list, Identification
from src.optimize import (
    optimize, refine_molecules,
    random_mutation_by_group, prepare_pred_data
)


def main(config):
    T_back = config["xclass"].get("tBack", 0.)
    obs_data = load_preprocess(config["file_spec"], T_back)

    config_opt = config["opt_combine"]
    config_xclass = config["xclass"]
    pool = Pool(config_opt["n_process"])
    save_dir = config["save_dir"]

    # Read pred data
    pred_data_dict = {}
    for fname in Path(save_dir).glob("*.pickle"):
        pred_data = pickle.load(open(fname, "rb"))
        pred_data_dict[pred_data["name"]] = pred_data

    blending_list = derive_blending_list(
        obs_data, pred_data_dict.values(), T_back, **config_opt["kwargs"])
    idn = Identification(obs_data, T_back, **config_opt["kwargs"])

    name, inds_peak, state = init_state(blending_list, pred_data_dict, idn, config_xclass)
    idx_pop = 0
    i_group = 0
    while len(blending_list) != 0:
        blending_list.pop(idx_pop)
        for idx_pop, (name, inds_obs) in enumerate(blending_list):
            if not inds_peak.isdisjoint(inds_obs):
                inds_peak.update(inds_obs)
                params_0, mol_list, segments, include_list = combine(
                    state, pred_data_dict[name], config_xclass
                )
                model = create_fitting_model(
                    obs_data=obs_data,
                    mol_list=mol_list,
                    include_list=include_list,
                    config_xclass=config_xclass,
                    config_opt=config_opt,
                )
                initial_pos = derive_initial_pos(
                    model.func.pm, params_0, model.bounds,
                    n_swarm=config_opt["kwargs_opt"]["nswarm"],
                    prob=config_opt["prob_mutation"]
                )
                config_opt["kwargs_opt"]["initial_pos"] = initial_pos
                res_dict = optimize(model, name, segments, config_opt, pool)

                state.update(
                    params=res_dict["params_best"],
                    mol_list=mol_list,
                    segments=segments,
                    include_list=include_list
                )
                save_name = Path(save_dir) \
                    / Path("combine") \
                    / Path("group_{}_n{}.pickle".format(i_group, len(mol_list)))
                pickle.dump(res_dict, open(save_name, "wb"))
                break
        else:
            mol_list = state["mol_list"]
            if len(mol_list) == 1:
                res_dict = pred_data_dict[name]
            save_name = Path(save_dir) \
                / Path("combine") \
                / Path("group_{}_n{}.pickle".format(i_group, len(mol_list)))
            pickle.dump(res_dict, open(save_name, "wb"))

            if len(blending_list) > 0:
                i_group += 1
                name, inds_peak, state = init_state(blending_list, pred_data_dict, idn, config_xclass)
                idx_pop = 0


def init_state(blending_list, pred_data_dict, idn, config_xclass):
    name, inds_peak = blending_list[0]
    pred_data = pred_data_dict[name]
    wrapper = create_wrapper_from_config(None, pred_data["mol_list"], config_xclass)
    mol_list, include_list, params = filter_moleclues(
        idn=idn,
        pm=wrapper.pm,
        segments=pred_data["segments"],
        include_list=pred_data["include_list"],
        T_pred_data=pred_data["T_pred"],
        trans_data=pred_data["trans_dict"],
        params=pred_data["params_best"]
    )
    state = {
        "idn": idn,
        "params": params,
        "mol_list": mol_list,
        "segments": pred_data["segments"],
        "include_list": include_list,
    }
    return name, inds_peak, state


def combine(state, pred_data, config_xclass):
    wrapper = create_wrapper_from_config(None, pred_data["mol_list"], config_xclass)
    mol_list, include_list, params = filter_moleclues(
        idn=state["idn"],
        pm=wrapper.pm,
        segments=pred_data["segments"],
        include_list=pred_data["include_list"],
        T_pred_data=pred_data["T_pred"],
        trans_data=pred_data["trans_dict"],
        params=pred_data["params_best"]
    )

    params_list = [state["params"], params]
    mol_list_list = [state["mol_list"], mol_list]
    segments_list = [state["segments"], pred_data["segments"]]
    include_list_list = [state["include_list"], include_list]

    params_new, mol_list_new, segments_new, include_list_new = refine_molecules(
        params_list, mol_list_list, segments_list, include_list_list, config_xclass
    )
    return params_new, mol_list_new, segments_new, include_list_new


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