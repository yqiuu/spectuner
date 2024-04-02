import re
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
try:
    from xclass import task_ListDatabase
except ImportError:
    warnings.warn("XCLASS is not installed.")

from .atoms import MolecularDecomposer


def select_molecules(FreqMin, FreqMax, ElowMin, ElowMax,
                     molecules, elements, base_only,
                     iso_list=None, exclude_list=None, rename_dict=None):
    if molecules is None:
        molecules = []
        skip = True
    else:
        skip = False
    if iso_list is None:
        iso_list = []

    normal_dict = group_by_normal_form(
        FreqMin, FreqMax, ElowMin, ElowMax, elements, exclude_list, rename_dict
    )
    mol_dict, _ = replace_with_master_name(
        normal_dict, molecules, base_only, iso_list
    )
    if skip:
        return mol_dict

    mol_dict_ret = {}
    for name in molecules:
        if name in mol_dict:
            mol_dict_ret[name] = mol_dict[name]
    return mol_dict_ret


def select_molecules_multi(freq_data, ElowMin, ElowMax,
                           elements, molecules, base_only,
                           iso_list=None, exclude_list=None, rename_dict=None):
    if iso_list is None:
        iso_list = []

    normal_dict_list = []
    for freq in freq_data:
        normal_dict_list.append(group_by_normal_form(
            FreqMin=freq[0],
            FreqMax=freq[-1],
            ElowMin=ElowMin,
            ElowMax=ElowMax,
            elements=elements,
            moleclues=molecules,
            exclude_list=exclude_list,
            rename_dict=rename_dict
        ))

    # Merge all normal dict from different segment
    normal_dict_all = defaultdict(list)
    for normal_dict in normal_dict_list:
        for key, name_list in normal_dict.items():
            normal_dict_all[key].extend(name_list)
    # Remove duplicated moleclues in the normal dict
    for key in list(normal_dict_all.keys()):
        tmp = list(set(normal_dict_all[key]))
        tmp.sort()
        normal_dict_all[key] = tmp

    mol_list, master_name_dict \
        = replace_with_master_name(normal_dict_all, base_only, iso_list)

    incldue_dict = defaultdict(lambda: [[] for _ in range(len(freq_data))])
    for i_segment, normal_dict in enumerate(normal_dict_list):
        for name, iso_list in normal_dict.items():
            master_name = master_name_dict[name]
            if master_name is not None:
                incldue_dict[master_name][i_segment]= deepcopy(iso_list)
    incldue_dict = dict(incldue_dict)

    return mol_list, incldue_dict


def group_by_normal_form(FreqMin, FreqMax, ElowMin, ElowMax,
                         elements, moleclues, exclude_list, rename_dict):
    if exclude_list is None:
        exclude_list = []
    if rename_dict is None:
        rename_dict = {}

    contents = task_ListDatabase.ListDatabase(
        FreqMin, FreqMax, ElowMin, ElowMax,
        SelectMolecule=[], OutputDevice="quiet"
    )

    mol_names = set()
    for item in contents:
        mol_names.add(item.split()[0])
    mol_names = list(mol_names)
    mol_names.sort()

    mol_dict = defaultdict(list)
    for name in mol_names:
        if name in exclude_list:
            continue
        tmp = name.split(";")
        # Ingore spin states
        if tmp[-1] == "ortho" or tmp[-1] == "para":
            continue
        mol_dict[";".join(tmp[:-1])].append(tmp[-1])

    mol_names = []
    for key, val in mol_dict.items():
        mol_names.append(";".join([key, val[0]]))

    if moleclues is not None:
        moleclues = [derive_normal_form(name, rename_dict)[0] for name in moleclues]

    # Filter elements and molecules
    elements = set(elements)
    normal_dict = defaultdict(list)
    for name in mol_names:
        fm_normal, atom_set = derive_normal_form(name, rename_dict)
        if len(atom_set - elements) == 0 and (moleclues is None or fm_normal in moleclues):
            normal_dict[fm_normal].append(name)
    return normal_dict


def derive_normal_form(mol_name, rename_dict):
    fm, *_ = mol_name.split(";")
    if fm in rename_dict:
        fm = rename_dict[fm]

    atom_dict = MolecularDecomposer(fm).ShatterFormula()
    atom_set = set(atom_dict.keys())

    pattern = r"-([0-9])([0-9])[-]?"
    fm = re.sub(pattern, "", fm)
    for pattern in re.findall("[A-Z][a-z]?\d", fm):
        fm = fm.replace(pattern, pattern[:-1]*int(pattern[-1]))
    fm = fm.replace("D", "H")
    return fm, atom_set


def replace_with_master_name(normal_dict, base_only, iso_list):
    mol_dict = defaultdict(list)
    master_name_dict = {}
    for normal_name, name_list in normal_dict.items():
        name_list.sort()
        master_name = select_master_name(name_list, base_only)
        master_name_dict[normal_name] = master_name
        if master_name is None:
            continue
        for name in name_list:
            if base_only and name.split(";")[1] != "v=0" \
                and name not in iso_list:
                continue
            if name == master_name:
                mol_dict[master_name]
            else:
                mol_dict[master_name].append(name)
    mol_list = []
    for idx, (name, mols) in enumerate(mol_dict.items()):
        item = {"id": idx, "root": name}
        mols_new = [name]
        mols_new.extend(mols)
        item["molecules"] = mols_new
        mol_list.append(item)
    return mol_list, master_name_dict


def select_master_name(name_list, base_only):
    master_name_list = []
    for name in name_list:
        is_master = True
        pattern = r"-([0-9])([0-9])[-]?"
        if re.search(pattern, name) is not None:
            is_master = False
        if "D" in name:
            is_master = False
        if name.split(";")[1] != "v=0":
            is_master = False
        if is_master:
            master_name_list.append(name)
    if len(master_name_list) == 0:
        if base_only:
            master_name = None
        else:
            master_name = name_list[0]
    elif len(master_name_list) == 1:
        master_name = master_name_list[0]
    else:
        raise ValueError("Multiple master name", master_name_list)
    return master_name


def derive_median_frac_threshold(obs_data, median_frac):
    T_obs = np.concatenate([spec[:, 1] for spec in obs_data])
    T_max = T_obs.max()
    T_median = np.median(T_obs)
    T_thr = T_median + median_frac*(T_max - T_median)
    return T_thr