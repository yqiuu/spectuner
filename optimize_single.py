import sys
import shutil
import yaml
import pickle
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from swing import ParticleSwarm

from src.xclass_wrapper import extract_line_frequency
from src.fitting_model import create_fitting_model_extra
from src.algorithms import select_molecules


def main(config):
    spec_obs = np.loadtxt(config["file_spec"])
    spec_obs = spec_obs[::-1] # Make freq ascending

    pool = Pool(config["opt_single"]["n_process"])
    FreqMin = spec_obs[0, 0]
    FreqMax = spec_obs[-1, 0]
    ElowMin = 0
    ElowMax = 2000.
    mol_names, iso_dict = select_molecules(
        FreqMin, FreqMax, ElowMin, ElowMax,
        config["molecules"], config["elements"]
    )
    for name in mol_names:
        iso_dict_sub = {}
        if name in iso_dict:
            iso_dict_sub[name] = iso_dict[name]
        optimize(spec_obs, name, iso_dict_sub, config, pool)


def optimize(spec_obs, mol_name, iso_dict, config, pool):
    config_opt = config["opt_single"]
    model = create_fitting_model_extra(
        spec_obs, [mol_name], iso_dict,
        config["opt_single"]["loss_fn"], config, vLSR=0.
    )
    opt = ParticleSwarm(model, model.bounds, nswarm=config_opt["n_swarm"], pool=pool)
    opt.swarm(config_opt["n_cycle"])

    params_best = model.derive_params(opt.pos_global_best)
    spectrum, _, trans, _, job_dir = model.func.call_full_output(params_best)
    T_pred = spectrum[:, 1]
    trans_dict = extract_line_frequency(trans)
    shutil.rmtree(job_dir)

    save_dict = {
        "name": mol_name,
        "cost_best": opt.cost_global_best,
        "params_best": opt.pos_global_best,
        "T_pred": T_pred,
        "trans_dict": trans_dict,
    }
    if config_opt["save_history"]:
        save_dict["history"] = opt.memo
    save_dir = Path(config["save_dir"])
    pickle.dump(save_dict, open(save_dir/Path("{}.pickle".format(mol_name)), "wb"))


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    main(config)