import numpy as np

from .xclass_wrapper import create_wrapper_from_config


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