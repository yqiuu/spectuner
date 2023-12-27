import sys
import yaml
import pickle
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool

from swing import ParticleSwarm

from src.xclass_wrapper import extract_line_frequency
from src.preprocess import load_spectra
from src.fitting_model import create_fitting_model_extra


def main(config):
    obs_data, mol_dict, mol_list, segment_dict = load_spectra(config)
    pool = Pool(config["opt_single"]["n_process"])
    for name in mol_list:
        if segment_dict is None:
            obs_data_sub = obs_data
            mol_dict_sub = {name: mol_dict[name]}
            segments = None
        else:
            obs_data_sub = []
            mol_dict_sub = defaultdict(list)
            for idx in segment_dict[name]:
                obs_data_sub.append(obs_data[idx])
                mol_dict_sub[name].extend(mol_dict[idx][name])
            tmp = list(set(mol_dict_sub[name]))
            tmp.sort()
            mol_dict_sub[name] = tmp
            segments = segment_dict[name]
        ret_dict = optimize(obs_data_sub, mol_dict_sub, segments, config, pool)
        save_dir = Path(config["save_dir"])
        pickle.dump(ret_dict, open(save_dir/Path("{}.pickle".format(name)), "wb"))


def optimize(obs_data, mol_dict, segments, config, pool):
    config_opt = config["opt_single"]
    model = create_fitting_model_extra(
        obs_data, mol_dict,
        config["xclass"], config["opt_single"],
    )
    opt = ParticleSwarm(model, model.bounds, nswarm=config_opt["n_swarm"], pool=pool)
    opt.swarm(config_opt["n_cycle"])

    T_pred_data, trans_data, job_dir_data = model.call_func(opt.pos_global_best)
    if isinstance(job_dir_data, str):
        T_pred_data = [T_pred_data]
        trans_dict = [extract_line_frequency(trans_data)]
    else:
        trans_dict = [extract_line_frequency(trans) for trans in trans_data]

    # Get the first item in mol_dict
    for mol_name in mol_dict:
        break
    ret_dict = {
        "name": mol_name,
        "cost_best": opt.cost_global_best,
        "params_best": opt.pos_global_best,
        "freq": model.freq_data,
        "T_pred": T_pred_data,
        "trans_dict": trans_dict,
        "segments": segments
    }
    if config_opt["save_history"]:
        ret_dict["history"] = opt.memo
    return ret_dict


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    main(config)