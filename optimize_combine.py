import sys
import yaml
import pickle
from pathlib import Path
from multiprocessing import Pool

import numpy as np

from src.preprocess import load_preprocess
from src.fitting_model import create_fitting_model_extra
from src.xclass_wrapper import create_wrapper_from_config
from src.algorithms import filter_moleclues, Identification
from src.optimize import (
    optimize, refine_molecules,
    shrink_bounds, random_mutation_by_group
)


def main(config):
    T_back = config["xclass"].get("tBack", 0.)
    obs_data = load_preprocess(config["file_spec"], T_back)

    config_opt = config["opt_combine"]
    pool = Pool(config_opt["n_process"])

    save_dir = config["save_dir"]
    model, params_0, segments = create_model(
        obs_data, save_dir, config["xclass"], config_opt
    )
    name = "combine"
    initial_pos = derive_initial_pos(
        model.func.pm, params_0, model.bounds,
        n_swarm=config_opt["kwargs_opt"]["nswarm"],
        prob=config_opt["prob_mutation"]
    )
    config_opt["kwargs_opt"]["initial_pos"] = initial_pos
    ret_dict = optimize(model, name, segments, config_opt, pool)
    pickle.dump(ret_dict, open(Path(save_dir)/Path("{}.pickle".format(name)), "wb"))


def create_model(obs_data, save_dir, config_xclass, config_opt):
    T_back = config_xclass.get("tBack", 0.)
    kwargs = {"prominence": config_opt["kwargs"]["prominence"]}
    try:
        kwargs["rel_height"] = config_opt["kwargs"]["rel_height"]
    except KeyError:
        pass

    idn = Identification(obs_data, T_back, **kwargs)
    params_list = []
    mol_list_list = []
    segments_list = []
    include_list_list = []
    for fname in Path(save_dir).glob("*.pickle"):
        if str(fname.name).startswith("combine"):
            continue

        data = pickle.load(open(fname, "rb"))
        wrapper = create_wrapper_from_config(None, data["mol_list"], config_xclass)
        mol_list, include_list, params = filter_moleclues(
            idn=idn,
            pm=wrapper.pm,
            segments=data["segments"],
            include_list=data["include_list"],
            T_pred_data=data["T_pred"],
            trans_data=data["trans_dict"],
            params=data["params_best"]
        )

        params_list.append(params)
        mol_list_list.append(mol_list)
        segments_list.append(data["segments"])
        include_list_list.append(include_list)

    params_new, mol_dict_new, segments_new, include_list_new = refine_molecules(
        params_list, mol_list_list, segments_list, include_list_list, config_xclass
    )
    model = create_fitting_model_extra(
        obs_data=obs_data,
        mol_dict=mol_dict_new,
        include_list=include_list_new,
        config_xclass=config_xclass,
        config_opt=config_opt,
    )
    """
    bounds_new = shrink_bounds(
        pm=model.func.pm,
        params=params_new,
        bounds_mol=config_opt["bounds_mol"],
        delta_mol=config_opt["delta_mol"],
        bounds_iso=config_opt["bounds_iso"],
        delta_iso=config_opt["delta_iso"],
        bounds_misc=config_opt.get("bounds_misc", None)
    )
    model.bounds = bounds_new
    """
    return model, params_new, segments_new


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