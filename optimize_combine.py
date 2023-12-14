import sys
import yaml
import pickle
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from swing import ParticleSwarm

from src.xclass_wrapper import load_molfit_info
from src.fitting_model import create_fitting_model


def main(config):
    spec_obs = np.loadtxt(config["file_spec"])
    spec_obs = spec_obs[::-1] # Make freq ascending

    config_opt = config["opt_combine"]
    pool = Pool(config_opt["n_process"])
    mol_names, bounds = load_molfit_info(config["file_molfit"])
    bounds = bounds.reshape(-1, 2)
    model = create_fitting_model(spec_obs, mol_names, bounds, config, vLSR=0.)
    initial_pos = refine(model, config)
    opt = ParticleSwarm(
        model, model.bounds, nswarm=config_opt["n_swarm"],
        pool=pool, initial_pos=initial_pos
    )
    save_dir = Path(config["save_dir"])
    for i_cycle in range(config_opt["n_cycle"]):
        opt.swarm(1)
        opt.save_checkpoint(save_dir/Path(f"ckpt_{i_cycle%2}.pickle"))
        pickle.dump(opt.memo,open(save_dir/Path(f"history_{i_cycle%2}.pickle"), "wb"))


def refine(model, config):
    mol_names = model.func.mol_names
    n_param_per_mol = model.func.n_param_per_mol
    n_mol_param = model.func.n_mol_param

    params_combine = load_params_combine(mol_names, config["save_dir"])
    params_mol = params_combine[:, :n_param_per_mol]

    delta = np.array(config["refine"]["delta"])
    bounds = model.bounds
    bounds_mol = bounds[:n_mol_param].reshape(-1, n_param_per_mol, 2)
    bounds_mol_new = shrink_bounds(params_mol, bounds_mol, delta)
    bounds_misc = bounds[n_mol_param:]
    bounds_new = np.vstack([bounds_mol_new.reshape(-1, 2), bounds_misc])
    model.bounds = bounds_new

    initial_pos = [new_init_pos(params_mol, bounds_mol, bounds_misc, 0)]
    n_replace = len(params_mol)//2
    for _ in range(config["opt_combine"]["n_swarm"] - 1):
        initial_pos.append(new_init_pos(params_mol, bounds_mol, bounds_misc, n_replace))
    initial_pos = np.vstack(initial_pos)
    return initial_pos


def shrink_bounds(params_mol, bounds_mol, delta):
    # bounds_mol (N, X, 2)
    bounds_new = np.zeros_like(bounds_mol)
    for i_mol in range(len(bounds_mol)):
        lb = np.maximum(params_mol[i_mol] - .5*delta, bounds_mol[i_mol, :, 0])
        ub = np.minimum(params_mol[i_mol] + .5*delta, bounds_mol[i_mol, :, 1])
        bounds_new[i_mol, :, 0] = lb
        bounds_new[i_mol, :, 1] = ub
    return bounds_new


def new_init_pos(params_mol, bounds_mol, bounds_misc, n_replace):
    # bounds_mol (N, X, 2)
    # bounds_misc (N, 2)
    params_mol = params_mol.copy()
    params_misc = np.random.uniform(bounds_misc[:, 0], bounds_misc[:, 1])
    inds = np.random.choice(len(params_mol), n_replace, replace=False)
    for idx in inds:
        params_mol[idx] = np.random.uniform(bounds_mol[idx, :, 0], bounds_mol[idx, :, 1])
    params_new = np.append(np.ravel(params_mol), params_misc)
    return params_new


def load_params_combine(mol_names, dirname):
    params = []
    for name in mol_names:
        fname = Path(dirname)/Path("{}.npy".format(name))
        params.append(np.load(fname))
    params = np.vstack(params)
    return params


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    main(config)