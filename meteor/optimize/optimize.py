from multiprocessing import Pool

import numpy as np
from swing import ParticleSwarm, ArtificialBeeColony
from tqdm import trange

from ..xclass_wrapper import extract_line_frequency


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

    n_cycle_min = config_opt["n_cycle_min"] \
        + (len(model.bounds) - 5)*config_opt["n_cycle_dim"]
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

        if i_cycle > 1:
            rate = compute_rate(opt)
            if rate < tol_stop:
                n_stuck += 1
            else:
                n_stuck = 0

        if n_stuck >= n_stop and i_cycle + 1 >= n_cycle_min:
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


def create_pool(n_process, use_mpi):
    if use_mpi:
        from mpi4py.futures import MPIPoolExecutor
        return MPIPoolExecutor(n_process)
    else:
        return Pool(n_process)


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