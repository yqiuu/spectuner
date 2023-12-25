import sys
import shutil
import yaml
import pickle
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from swing import ParticleSwarm

from src.xclass_wrapper import extract_line_frequency
from src.preprocess import preprocess_spectrum
from src.fitting_model import create_fitting_model_extra
from src.algorithms import select_molecules, select_molecules_multi


def main(config):
    ElowMin = 0
    ElowMax = 2000.
    temp_back = config["xclass"].get("tBack", 0.)
    if isinstance(config["file_spec"], str):
        obs_data = np.loadtxt(config["file_spec"])
        obs_data = preprocess_spectrum(obs_data, temp_back)
        FreqMin = obs_data[0, 0]
        FreqMax = obs_data[-1, 0]
        mol_dict = select_molecules(
            FreqMin, FreqMax, ElowMin, ElowMax,
            config["molecules"], config["elements"], config["base_only"]
        )
        segment_dict = None
        mol_list = list(mol_dict.keys())
    else:
        obs_data = [np.loadtxt(fname) for fname in config["file_spec"]]
        obs_data = [preprocess_spectrum(spec, temp_back) for spec in obs_data]
        mol_dict, segment_dict = select_molecules_multi(
            obs_data, ElowMin, ElowMax,
            config["molecules"], config["elements"], config["base_only"]
        )
        mol_list = list(segment_dict.keys())
    pool = Pool(config["opt_single"]["n_process"])
    for name in mol_list:
        if segment_dict is None:
            obs_data_sub = obs_data
            mol_dict_sub = {name: mol_dict[name]}
        else:
            obs_data_sub = []
            mol_dict_sub = defaultdict(list)
            for idx in segment_dict[name]:
                obs_data_sub.append(obs_data[idx])
                mol_dict_sub[name].extend(mol_dict[idx][name])
        ret_dict = optimize(obs_data, mol_dict_sub, config, pool)
        save_dir = Path(config["save_dir"])
        pickle.dump(ret_dict, open(save_dir/Path("{}.pickle".format(name)), "wb"))


def optimize(obs_data, mol_dict, config, pool):
    config_opt = config["opt_single"]
    model = create_fitting_model_extra(
        obs_data, mol_dict,
        config["xclass"], config["opt_single"],
    )
    opt = ParticleSwarm(model, model.bounds, nswarm=config_opt["n_swarm"], pool=pool)
    opt.swarm(config_opt["n_cycle"])

    T_pred_data, trans_data, job_dir_data = model.call_func(opt.pos_global_best)
    if isinstance(job_dir_data, str):
        trans_dict = extract_line_frequency(trans_data)
        shutil.rmtree(job_dir_data)
    else:
        trans_dict = [extract_line_frequency(trans) for trans in trans_data]
        for job_dir in job_dir_data:
            shutil.rmtree(job_dir)

    # Get the first item in mol_dict
    for mol_name in mol_dict:
        break
    ret_dict = {
        "name": mol_name,
        "cost_best": opt.cost_global_best,
        "params_best": opt.pos_global_best,
        "T_pred": T_pred_data,
        "trans_dict": trans_dict,
    }
    if config_opt["save_history"]:
        ret_dict["history"] = opt.memo
    return ret_dict


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    main(config)