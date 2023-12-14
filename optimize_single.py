import sys
import yaml
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from swing import ParticleSwarm

from src.xclass_wrapper import load_molfit_info
from src.fitting_model import create_fitting_model


def main(config):
    spec_obs = np.loadtxt(config["file_spec"])
    spec_obs = spec_obs[::-1] # Make freq ascending

    pool = Pool(config["opt_single"]["n_process"])
    mol_names, bounds = load_molfit_info(config["file_molfit"])
    for idx in range(len(mol_names)):
        optimize(spec_obs, mol_names[idx], bounds[idx], config, pool)


def optimize(spec_obs, mol_names, bounds, config, pool):
    config_opt = config["opt_single"]
    model = create_fitting_model(spec_obs, [mol_names], bounds, config, vLSR=0.)
    opt = ParticleSwarm(model, model.bounds, nswarm=config_opt["n_swarm"], pool=pool)
    opt.swarm(config_opt["n_cycle"])

    save_dir = Path(config["save_dir"])
    np.save(save_dir/Path("{}".format(mol_names)), opt.pos_global_best)


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    main(config)