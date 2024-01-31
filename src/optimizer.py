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