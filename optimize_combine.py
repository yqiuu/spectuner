import sys
import shutil
import yaml
import pickle
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from swing import ParticleSwarm, ArtificialBeeColony

from src.fitting_model import create_fitting_model_extra
from src.algorithms import select_molecules, identify_single_score, identify_combine


def main(config):
    spec_obs = np.loadtxt(config["file_spec"])
    spec_obs = spec_obs[::-1] # Make freq ascending

    config_opt = config["opt_combine"]
    pool = Pool(config_opt["n_process"])
    FreqMin = spec_obs[0, 0]
    FreqMax = spec_obs[-1, 0]
    ElowMin = 0
    ElowMax = 2000.
    mol_dict = select_molecules(
        FreqMin, FreqMax, ElowMin, ElowMax,
        config["molecules"], config["elements"]
    )

    # Refine
    mol_dict, params_mol, params_iso \
        = refine_molecules(spec_obs, mol_dict, config)
    if len(mol_dict) == 0:
        raise ValueError("No molecule is selected.")

    model = create_fitting_model_extra(
        spec_obs, mol_dict,
        config["xclass"], config["opt_combine"], vLSR=0.
    )
    bounds_mol, bounds_iso = shrink_bounds(
        mol_dict.keys(), params_mol, params_iso,
        config["opt_combine"], config["refine"]
    )

    pm = model.func.pm
    bounds_misc = model.bounds[pm.inds_misc_param]

    bounds = np.vstack([bounds_mol, bounds_iso, bounds_misc])
    model.bounds = bounds

    args = pm, params_mol, params_iso, bounds_mol, bounds_iso, bounds_misc
    initial_pos = [new_init_pos(*args, n_replace=0)]
    n_replace = pm.n_mol//2
    for _ in range(config["opt_combine"]["n_swarm"] - 1):
        initial_pos.append(new_init_pos(*args, n_replace))
    initial_pos = np.vstack(initial_pos)

    save_dir = Path(config_opt["save_dir"])

    # Save
    model_info = {
        "mol_dict": mol_dict,
        "bounds": model.bounds
    }
    pickle.dump(model_info, open(save_dir/Path("model_info.pickle"), "wb"))

    #
    opt = ParticleSwarm(
        model, model.bounds, nswarm=config_opt["n_swarm"],
        pool=pool, initial_pos=initial_pos
    )
    for i_cycle in range(config_opt["n_cycle"]):
        opt.swarm(1)
        opt.save_checkpoint(save_dir/Path(f"ckpt_{i_cycle%2}.pickle"))
        pickle.dump(opt.memo,open(save_dir/Path(f"history_{i_cycle%2}.pickle"), "wb"))

    params_best = model.derive_params(opt.pos_global_best)
    spectrum, _, _, _, job_dir = model.func.call_full_output(params_best)
    shutil.move(job_dir, str(save_dir))
    job_dir = str(Path(save_dir)/Path(job_dir).name)
    save_dict = {
        "cost_best": opt.cost_global_best,
        "params_best": opt.pos_global_best,
        "T_pred": spectrum[:, 1],
        "job_dir": job_dir
    }
    pickle.dump(save_dict, open(save_dir/Path("combine.pickle"), "wb"))
    result = identify_combine(job_dir, mol_dict, spec_obs, **config["identify_combine"])
    pickle.dump(result, open(save_dir/Path("result.pickle"), "wb"))



def refine_molecules(spec_obs, mol_dict, config):
    mol_dict_new = {}
    params_mol = []
    params_iso = []

    for name, iso_list in mol_dict.items():
        fname = Path(config["save_dir"])/Path(f"{name}.pickle")
        data = pickle.load(open(fname, "rb"))

        mol_dict_sub = {name: iso_list}
        model = create_fitting_model_extra(
            spec_obs, mol_dict_sub,
            config["xclass"], config["opt_combine"], vLSR=0.
        )

        is_accepted = identify_single_score(
            spec_obs[:, 1], data["T_pred"], spec_obs[:, 0], data["trans_dict"],
            config["T_thr"],
        )
        if is_accepted:
            mol_dict_new[name] = mol_dict[name]
            pm = model.func.pm
            params = data["params_best"]
            params_mol.append(params[pm.inds_mol_param])
            params_iso.append(params[pm.inds_iso_param])
    if len(params_mol) == 0:
        params_mol = np.zeros(0)
        params_iso = np.zeros(0)
    else:
        params_mol = np.concatenate(params_mol)
        params_iso = np.concatenate(params_iso)

    return mol_dict_new, params_mol, params_iso


def shrink_bounds(mol_names, params_mol, params_iso, config_opt, config_refine):
    bounds_mol = np.tile(config_opt["bounds_mol"], (len(mol_names), 1))
    bounds_mol_new = np.zeros_like(bounds_mol)
    delta_mol = np.repeat(config_refine["delta_mol"], len(mol_names))
    # Set lower bounds
    bounds_mol_new[:, 0] = np.maximum(params_mol - .5*delta_mol, bounds_mol[:, 0])
    # Set upper bounds
    bounds_mol_new[:, 1] = np.minimum(params_mol + .5*delta_mol, bounds_mol[:, 1])

    bounds_iso = np.tile(config_opt["bounds_iso"], (len(params_iso), 1))
    bounds_iso_new = np.zeros_like(bounds_iso)
    delta_iso = np.full(len(params_iso), config_refine["delta_iso"])
    # Set lower bounds
    bounds_iso_new[:, 0] = np.maximum(params_iso - .5*delta_iso, bounds_iso[:, 0])
    # Set upper bounds
    bounds_iso_new[:, 1] = np.minimum(params_iso + .5*delta_iso, bounds_iso[:, 1])
    return bounds_mol_new, bounds_iso_new


def new_init_pos(pm, params_mol, params_iso, bounds_mol, bounds_iso, bounds_misc, n_replace):
    params_mol = params_mol.copy()
    params_iso = params_iso.copy()
    params_misc = np.random.uniform(bounds_misc[:, 0], bounds_misc[:, 1])
    mol_names = np.random.choice(pm.n_mol, n_replace, replace=False)
    for name in mol_names:
        inds = pm.get_mol_slice(name)
        params_mol[inds] = np.random.uniform(bounds_mol[inds, 0], bounds_mol[inds, 1])
        inds = pm.get_iso_slice(name)
        if inds is not None:
            params_iso[inds] = np.random.uniform(bounds_iso[inds, 0], bounds_iso[inds, 1])
    params_new = np.concatenate([params_mol, params_iso, params_misc])
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