import numpy as np
from swing import ParticleSwarm, ArtificialBeeColony
from tqdm import trange

from .xclass_wrapper import (
    create_wrapper_from_config, extract_line_frequency, MoleculeStore
)


def optimize(model, config_opt, pool):
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

    n_cycle_min = config_opt["n_cycle_min"]
    n_cycle_max = config_opt["n_cycle_max"]
    n_stop = config_opt["n_stop"]
    tol_stop = config_opt["tol_stop"]

    def compute_rate(opt):
        history = opt.memo["cost"]
        return abs(history[-2] - history[-1])/abs(history[-2])

    pos_all = []
    cost_all = []
    n_stuck = 0
    for i_cycle in trange(n_cycle_max):
        for data in opt.swarm(niter=1, progress_bar=False).values():
            if save_all:
                pos_all.append(data["pos"])
                cost_all.append(data["cost"])

        if i_cycle + 1 >= max(2, n_cycle_min):
            rate = compute_rate(opt)
            if rate < tol_stop:
                n_stuck += 1
            else:
                n_stuck = 0

            if n_stuck == n_stop:
                break

    if len(pos_all) != 0:
        pos_all = np.vstack(pos_all)
        cost_all = np.concatenate(cost_all)

    T_pred_data, trans_data = prepare_pred_data(model, opt.pos_global_best)

    ret_dict = {
        "mol_store": model.mol_store,
        "freq": model.freq_data,
        "T_pred": T_pred_data,
        "trans_dict": trans_data,
        "cost_best": opt.cost_global_best,
        "params_best": opt.pos_global_best,
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
    if config_opt.get("save_T_target", False):
        ret_dict["T_target"] = model.T_obs_data
    return ret_dict


def prepare_pred_data(model, pos):
    T_pred_data, trans_data, _, job_dir_data = model.call_func(pos)
    if isinstance(job_dir_data, str):
        T_pred_data = [T_pred_data]
        trans_data_ret = [extract_line_frequency(trans_data)]
        return T_pred_data, trans_data_ret

    trans_data_ret = []
    for trans in trans_data:
        if trans is None:
            trans_data_ret.append(None)
        else:
            trans_data_ret.append(extract_line_frequency(trans))
    return T_pred_data, trans_data_ret


def combine_mol_stores(mol_store_list, params_list, config_slm):
    mol_list = []
    include_list = [[] for _ in range(len(mol_store_list[0].include_list))]
    for mol_store in mol_store_list:
        mol_list.extend(mol_store.mol_list)
        for in_list_new, in_list in zip(include_list, mol_store.include_list):
            in_list_new.extend(in_list)
    mol_store_new = MoleculeStore(mol_list, include_list, mol_store_list[0].scaler)

    params_mol = []
    params_den = []
    for mol_store, params in zip(mol_store_list, params_list):
        pm = mol_store.create_parameter_manager(config_slm)
        params_mol.append(pm.get_all_mol_params(params))
        params_den.append(pm.get_all_den_params(params))
    params_mol = np.concatenate(params_mol)
    params_den = np.concatenate(params_den)
    params_new = np.append(params_mol, params_den)
    return mol_store_new, params_new


def shrink_bounds(pm, params, bounds_mol, delta_mol, bounds_iso, delta_iso, bounds_misc):
    params_mol = pm.get_all_mol_params(params)
    bounds_mol = np.tile(bounds_mol, (pm.n_mol, 1))
    bounds_mol_new = np.zeros_like(bounds_mol)
    delta_mol = np.repeat(delta_mol, pm.n_mol)
    # Set lower bounds
    bounds_mol_new[:, 0] = np.maximum(params_mol - .5*delta_mol, bounds_mol[:, 0])
    # Set upper bounds
    bounds_mol_new[:, 1] = np.minimum(params_mol + .5*delta_mol, bounds_mol[:, 1])

    params_iso = pm.get_all_iso_params(params)
    bounds_iso = np.tile(bounds_iso, (pm.n_iso_param, 1))
    bounds_iso_new = np.zeros_like(bounds_iso)
    delta_iso = np.full(len(params_iso), delta_iso)
    # Set lower bounds
    bounds_iso_new[:, 0] = np.maximum(params_iso - .5*delta_iso, bounds_iso[:, 0])
    # Set upper bounds
    bounds_iso_new[:, 1] = np.minimum(params_iso + .5*delta_iso, bounds_iso[:, 1])

    bounds_new = np.vstack([bounds_mol_new, bounds_iso_new])
    if pm.n_misc_param > 0:
        bounds_new = np.vstack([bounds_new, np.atleast_2d(bounds_misc)])
    return bounds_new


def random_mutation(params, bounds, prob, rstate=None):
    if rstate is None:
        rstate = np.random
    params_new = np.zeros_like(params)
    for i_p in range(len(params)):
        if rstate.rand() < prob:
            lower, upper = bounds[i_p]
            val = lower + (upper - lower)*rstate.rand()
        else:
            val = params[i_p]
        params_new[i_p] = val
    return params_new


def random_mutation_by_group(pm, params, bounds, prob=0.4, rstate=None):
    """Perturb the parameters by a group of molecules.

    Args:
        pm (ParameterManager): Parameter Manager.
        params (array): Parameters
        bounds (array): Bounds
        prob (float, optional): Mutation probability. The code will perturb at
            least one group unless the probability is 0. Defaults to 0.4.
        rstate (RandomState, optional): RNG.

    Returns:
        _type_: _description_
    """
    if rstate is None:
        rstate = np.random

    params_mol, params_den, params_misc = pm.split_params(params, need_reshape=False)
    lb_mol, lb_den, _ = pm.split_params(bounds[:, 0], need_reshape=False)
    ub_mol, ub_den, _ = pm.split_params(bounds[:, 1], need_reshape=False)

    params_mol = params_mol.copy()
    params_den = params_den.copy()
    n_replace = int(prob*pm.n_mol)
    if n_replace == 0 and prob > 0:
        n_replace = 1
    id_list = np.random.choice(pm.id_list, n_replace, replace=False)
    for key in id_list:
        inds = pm.get_mol_slice(key)
        params_mol[inds] = rstate.uniform(lb_mol[inds], ub_mol[inds])
        inds = pm.get_den_slice(key)
        if inds is not None:
            params_den[inds] = np.random.uniform(lb_den[inds], ub_den[inds])
    params_new = np.concatenate([params_mol, params_den, params_misc])
    return params_new