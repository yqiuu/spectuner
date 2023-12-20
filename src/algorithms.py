import re
from pathlib import Path
from collections import defaultdict

import numpy as np

from .atoms import MolecularDecomposer
from .xclass_wrapper import task_ListDatabase, extract_line_frequency


def select_molecules(FreqMin, FreqMax, ElowMin, ElowMax, molecules, elements):
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
        tmp = name.split(";")
        mol_dict[";".join(tmp[:-1])].append(tmp[-1])

    mol_names = []
    for key, val in mol_dict.items():
        mol_names.append(";".join([key, val[0]]))

    # Filter elements
    elements = set(elements)
    normal_dict = defaultdict(list)
    for name in mol_names:
        fm_normal, atom_set = derive_normal_form(name)
        if len(atom_set - elements) == 0:
            normal_dict[fm_normal].append(name)

    mol_dict = defaultdict(list)
    for name_list in normal_dict.values():
        master_name = []
        for name in name_list:
            is_master = True
            pattern = r"-([0-9])([0-9])[-]?"
            if re.search(pattern, name) is not None:
                is_master = False
            if name.split(";")[1] != "v=0":
                is_master = False
            if is_master:
                master_name.append(name)
        if len(master_name) == 0:
            pass
        elif len(master_name) == 1:
            master_name = master_name[0]
            for name in name_list:
                if name == master_name:
                    mol_dict[master_name]
                else:
                    mol_dict[master_name].append(name)
        else:
            raise ValueError("Multiple master name", master_name)

    mol_names = []
    iso_dict = {}
    if molecules is None:
        for key, val in mol_dict.items():
            mol_names.append(key)
            if len(val) > 0:
                iso_dict[key] = val
        return mol_names, iso_dict

    for name in molecules:
        if name in mol_dict:
            mol_names.append(name)
            if len(mol_dict[name]) > 0:
                iso_dict[name] = mol_dict[name]
    return mol_names, iso_dict


def derive_normal_form(mol_name):
    fm, *_ = mol_name.split(";")
    atom_dict = MolecularDecomposer(fm).ShatterFormula()
    atom_set = set(atom_dict.keys())

    pattern = r"-([0-9])([0-9])[-]?"
    fm = re.sub(pattern, "", fm)
    for pattern in re.findall("[A-Z][a-z]?\d", fm):
        fm = fm.replace(pattern, pattern[:-1]*int(pattern[-1]))
    return fm, atom_set


def identify_single(T_obs, T_pred, freq, trans_dict, T_thr, tol=.1, return_full=False):
    line_freq = []
    for _, freq_list in trans_dict.items():
        line_freq.extend(freq_list)

    info = []
    freq_min = freq[0]
    freq_max = freq[-1]
    is_accepted = False
    for nu in line_freq:
        if nu < freq_min or nu > freq_max:
            continue

        idx = np.argmin(np.abs(freq - nu))
        err = np.abs((T_pred[idx] - T_obs[idx])/T_obs[idx])
        if err < tol and T_obs[idx] > T_thr:
            is_accepted = True
            info.append((nu, True))
        else:
            info.append((nu, False))

    if return_full:
        return is_accepted, info
    return is_accepted


def identify_single_score(T_obs, T_pred, freq, trans_dict, T_thr):
    line_freq = []
    for _, freq_list in trans_dict.items():
        line_freq.extend(freq_list)

    count_pos = 0
    count_neg = 0
    freq_min = freq[0]
    freq_max = freq[-1]
    for nu in line_freq:
        if nu < freq_min or nu > freq_max:
            continue

        idx = np.argmin(np.abs(freq - nu))
        if T_obs[idx] > T_thr:
            if T_pred[idx] > T_thr:
                count_pos += 1
        else:
            if T_pred[idx] > T_thr:
                count_neg += 1
    if count_neg >= 2:
        is_accepted = False
    elif count_pos > count_neg:
        is_accepted = True
    else:
        is_accepted = False
    return is_accepted


def identify_combine(job_dir, mol_names, iso_dict, spec_obs, T_thr, tol=.1):
    freq = spec_obs[:, 0]
    T_obs = spec_obs[:, 1]

    job_dir = Path(job_dir)
    transitions = open(job_dir/"transition_energies.dat").readlines()[1:]
    transitions = [line.split() for line in transitions]
    transitions = extract_line_frequency(transitions)

    # Get background temperature
    for line in open(job_dir/Path("xclass_spectrum.log")).readlines():
        if "Background Temperature" in line:
            temp_back = float(line.split()[-1].replace(r"\n", ""))

    def derive_fname(job_dir, name):
        name = name.replace("(", "_")
        name = name.replace(")", "_")
        name = name.replace(";", '_')
        return job_dir/Path("intensity__{}__comp__1.dat".format(name))

    mol_dict = {}
    spec_dict = {"freq": freq}
    for name in mol_names:
        fname = derive_fname(job_dir, name)
        T_pred = np.loadtxt(fname, skiprows=4)[:, 1]
        trans_dict = {name: transitions[name]}
        if name in iso_dict:
            for name_iso in iso_dict[name]:
                fname = derive_fname(job_dir, name_iso)
                T_pred += np.loadtxt(fname, skiprows=4)[:, 1]
                trans_dict[name_iso] = transitions[name_iso]
        T_pred += temp_back
        is_accepted = identify_single(T_obs, T_pred, freq, trans_dict, T_thr, tol)
        if is_accepted:
            mol_dict[name] = iso_dict.get(name, [])
            spec_dict[name] = T_pred
    return mol_dict, spec_dict