import sys
import yaml
import pickle
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from swing import ParticleSwarm, ArtificialBeeColony

from src.xclass_wrapper import extract_line_frequency
from src.preprocess import load_preprocess_select
from src.fitting_model import create_fitting_model_extra


def main(config):
    obs_data, mol_dict, segment_dict, include_dict = load_preprocess_select(config)
    pool = Pool(config["opt_single"]["n_process"])
    for name in mol_dict:
        model = create_model(name, obs_data, mol_dict, segment_dict, include_dict, config)
        segments = segment_dict[name]
        ret_dict = optimize(model, name, segments, config, pool)
        save_dir = Path(config["save_dir"])
        pickle.dump(ret_dict, open(save_dir/Path("{}.pickle".format(name)), "wb"))


def create_model(name, obs_data, mol_dict, segment_dict, include_dict, config):
    obs_data_sub = []
    for idx in segment_dict[name]:
        obs_data_sub.append(obs_data[idx])
    mol_dict_sub = {name: mol_dict[name]}
    include_list = include_dict[name]
    model = create_fitting_model_extra(
        obs_data_sub, mol_dict_sub, include_list,
        config["xclass"], config["opt_single"],
    )
    return model


def optimize(model, name, segments, config, pool):
    config_opt = config["opt_single"]

    opt_name = config_opt["optimizer"]
    if opt_name == "pso":
        cls_opt = ParticleSwarm
    elif opt_name == "abc":
        cls_opt = ArtificialBeeColony
    else:
        raise ValueError("Unknown optimizer: {}.".format(cls_opt))
    kwargs_opt = config_opt.get("kwargs_opt", {})
    opt = cls_opt(model, model.bounds, pool=pool, **kwargs_opt)
    save_all = config_opt.get("save_all", False)
    if save_all:
        pos_all = []
        cost_all = []
        for _ in range(config_opt["n_cycle"]):
            for data in opt.swarm(niter=1).values():
                pos_all.append(data["pos"])
                cost_all.append(data["cost"])
        pos_all = np.vstack(pos_all)
        cost_all = np.concatenate(cost_all)
    else:
        opt.swarm(config_opt["n_cycle"])

    T_pred_data, trans_data = prepare_pred_data(model, opt.pos_global_best)

    ret_dict = {
        "name": name,
        "freq": model.freq_data,
        "cost_best": opt.cost_global_best,
        "params_best": opt.pos_global_best,
        "T_pred": T_pred_data,
        "trans_dict": trans_data,
        "segments": segments,
        "mol_dict": model.func.pm.mol_dict,
        "include_list": model.include_list,
    }
    if config_opt.get("save_local_best", False):
        T_pred_data_local = []
        trans_data_local = []
        for pos in opt.pos_local_best:
            T_tmp, trans_tmp = prepare_pred_data(model, pos)
            T_pred_data_local.append(T_tmp)
            trans_data_local.append(trans_tmp)
        local_best = {
            "cost_best": opt.cost_local_best,
            "params_best": opt.pos_local_best,
            "T_pred": T_pred_data_local,
            "trans_dict": trans_data_local,
        }
        ret_dict["local_best"] = local_best
    if config_opt.get("save_history", False):
        ret_dict["history"] = opt.memo
    if save_all:
        ret_dict["pos_all"] = pos_all
        ret_dict["cost_all"] = cost_all
    return ret_dict


def prepare_pred_data(model, pos):
    T_pred_data, trans_data, job_dir_data = model.call_func(pos)
    if isinstance(job_dir_data, str):
        T_pred_data = [T_pred_data]
        trans_dict = [extract_line_frequency(trans_data)]
    else:
        trans_dict = [extract_line_frequency(trans) for trans in trans_data]
    return T_pred_data, trans_dict


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    main(config)