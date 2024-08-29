import sys
import pickle
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from swing import ParticleSwarm, ArtificialBeeColony
from tqdm import trange

from ..peaks import create_spans


def optimize(model, config_opt, pool):
    res_list = []
    for _ in range(config_opt["n_trail"]):
        res_list.append(optimize_sub(model, config_opt, pool))
    ret_dict = min(res_list, key=lambda x: x["cost_best"])
    save_all = config_opt.get("save_all", False)
    if save_all:
        cost_all = np.concatenate([res["cost_all"] for res in res_list])
        pos_all = np.vstack([res["pos_all"] for res in res_list])
        blob = []
        for res in res_list:
            blob.extend(res["blob"])
        ret_dict["cost_all"] = cost_all
        ret_dict["pos_all"] = pos_all
        ret_dict["blob"] = blob
    return ret_dict


def optimize_sub(model, config_opt, pool):
    opt_name = config_opt["optimizer"]
    if opt_name == "pso":
        cls_opt = ParticleSwarm
    elif opt_name == "abc":
        cls_opt = ArtificialBeeColony
    else:
        raise ValueError("Unknown optimizer: {}.".format(cls_opt))
    kwargs_opt = config_opt.get("kwargs_opt", {})
    blob = config_opt.get("blob", False)
    save_all = config_opt.get("save_all", False)
    opt = cls_opt(model, model.bounds, pool=pool, blob=blob, **kwargs_opt)

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
    blob = []
    n_stuck = 0
    for i_cycle in trange(n_cycle_max, file=sys.stdout):
        for data in opt.swarm(niter=1, progress_bar=False).values():
            if save_all:
                pos_all.append(data["pos"])
                cost_all.append(data["cost"])
                if data["blob"] is not None:
                    blob.extend(data["blob"])

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

    T_pred_data = model.sl_model(opt.pos_global_best)

    ret_dict = {
        "specie": model.sl_model.param_mgr.specie_list,
        "freq": model.freq_data,
        "T_pred": T_pred_data,
        "T_obs": model.T_obs_data,
        "T_base": model.T_base_data,
        "cost_best": opt.cost_global_best,
        "params_best": opt.pos_global_best,
    }
    if config_opt.get("save_local_best", False):
        T_pred_data_local = [model.sl_model(pos) for pos in opt.pos_local_best]
        local_best = {
            "cost_best": opt.cost_local_best,
            "params_best": opt.pos_local_best,
            "T_pred": T_pred_data_local,
        }
        ret_dict["local_best"] = local_best
    if config_opt.get("save_history", False):
        ret_dict["history"] = opt.memo
    if save_all:
        ret_dict["pos_all"] = pos_all
        ret_dict["cost_all"] = cost_all
        ret_dict["blob"] = blob
    return ret_dict


def create_pool(n_process, use_mpi):
    if use_mpi:
        from mpi4py.futures import MPIPoolExecutor
        return MPIPoolExecutor(n_process)
    else:
        return Pool(n_process)


def prepare_base_props(fname, config):
    if fname is not None:
        fname = Path(fname)
        fname = fname.with_name(f"identify_{fname.name}")
        res = pickle.load(open(fname, "rb"))
        T_base_data = res.get_T_pred()
        freqs_exclude = res.get_identified_lines()
        spans_include = create_spans(
            res.get_unknown_lines(), *config["opt"]["bounds"]["v_LSR"]
        )
        exclude_list = derive_exclude_list(res)

        id_offset = 0
        for key in res.mol_data:
            id_offset = max(id_offset, key)
        id_offset += 1
    else:
        T_base_data = None
        freqs_exclude = np.zeros(0)
        spans_include = np.zeros((0, 2))
        exclude_list = []
        id_offset = 0

    return {
        "T_base": T_base_data,
        "freqs_exclude": freqs_exclude,
        "spans_include": spans_include,
        "exclude_list": exclude_list,
        "id_offset": id_offset
    }


def derive_exclude_list(res):
    exclude_set = set()
    for sub_dict in res.mol_data.values():
        for key in sub_dict:
            exclude_set.add(key.split(";")[0])
    return list(exclude_set)


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