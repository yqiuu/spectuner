import numpy as np

from .xclass_wrapper import create_wrapper_from_config


def refine_molecules(params_list, mol_dict_list, include_list_list, config_xclass):
    params_mol = []
    params_iso = []
    mol_dict_ret = {}
    include_list_ret = [[] for _ in range(len(include_list_list[0]))]
    for params, mol_dict, include_list in zip(params_list, mol_dict_list, include_list_list):
        wrapper = create_wrapper_from_config(None, mol_dict, config_xclass)
        params_mol.append(wrapper.pm.get_all_mol_params(params))
        params_iso.append(wrapper.pm.get_all_iso_params(params))
        mol_dict_ret.update(mol_dict)
        for list_ret, list_new in zip(include_list_ret, include_list):
            list_ret.extend(list_new)
    params_mol = np.concatenate(params_mol)
    params_iso = np.concatenate(params_iso)
    params = np.append(params_mol, params_iso)
    return params, mol_dict_ret, include_list_ret


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