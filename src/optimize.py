import numpy as np
from swing import ParticleSwarm, ArtificialBeeColony

from .xclass_wrapper import create_wrapper_from_config, extract_line_frequency


def optimize(model, name, segments, config_opt, pool):
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
    n_cycle = config_opt["n_cycle"] + config_opt["cycle_factor"]*(len(model.bounds) - 5)
    if save_all:
        pos_all = []
        cost_all = []
        for _ in range(n_cycle):
            for data in opt.swarm(niter=1).values():
                pos_all.append(data["pos"])
                cost_all.append(data["cost"])
        pos_all = np.vstack(pos_all)
        cost_all = np.concatenate(cost_all)
    else:
        opt.swarm(n_cycle)

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
    if config_opt.get("save_T_target", False):
        ret_dict["T_target"] = model.T_obs_data
    return ret_dict


def prepare_pred_data(model, pos):
    T_pred_data, trans_data, job_dir_data = model.call_func(pos)
    if isinstance(job_dir_data, str):
        T_pred_data = [T_pred_data]
        trans_dict = [extract_line_frequency(trans_data)]
    else:
        trans_dict = [extract_line_frequency(trans) for trans in trans_data]
    return T_pred_data, trans_dict


def refine_molecules(params_list, mol_dict_list, segments_list, include_list_list, config_xclass):
    params_mol = []
    params_iso = []
    mol_dict_ret = {}
    for params, mol_dict in zip(params_list, mol_dict_list):
        wrapper = create_wrapper_from_config(None, mol_dict, config_xclass)
        params_mol.append(wrapper.pm.get_all_mol_params(params))
        params_iso.append(wrapper.pm.get_all_iso_params(params))
        mol_dict_ret.update(mol_dict)
    params_mol = np.concatenate(params_mol)
    params_iso = np.concatenate(params_iso)
    params = np.append(params_mol, params_iso)

    segments_ret = []
    for segments in segments_list:
        for idx in segments:
            if idx not in segments_ret:
                segments_ret.append(idx)
    n_segment = len(segments_ret)
    include_list_ret = [[] for _ in range(n_segment)]
    for segment, include_list in zip(segments_list, include_list_list):
        for i_segment, mol_list in zip(segment, include_list):
            include_list_ret[i_segment].extend(mol_list)
    return params, mol_dict_ret, segments_ret, include_list_ret


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