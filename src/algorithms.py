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

    if molecules is None:
        return mol_dict

    mol_dict_ret = {}
    for name in molecules:
        if name in mol_dict:
            mol_dict_ret[name] = mol_dict[name]
    return mol_dict_ret


def select_molecules_multi(obs_data, ElowMin, ElowMax, molecules, elements):
    mol_dict_list = []
    for spec in obs_data:
        mol_dict_list.append(select_molecules(
            FreqMin=spec[0, 0],
            FreqMax=spec[-1, 0],
            ElowMin=ElowMin,
            ElowMax=ElowMax,
            molecules=molecules,
            elements=elements
        ))
    segment_dict = defaultdict(list)
    for idx, mol_dict in enumerate(mol_dict_list):
        for name in mol_dict:
            segment_dict[name].append(idx)
    return mol_dict_list, segment_dict


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


def identify_single_score(T_obs, T_pred, freq, trans_dict, T_thr, tol):
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
            if np.abs(T_pred[idx] - T_obs[idx])/T_obs[idx] < tol:
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


def identify_combine(job_dir, mol_dict, spec_obs, T_thr, tol=.1):
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

    ret_dict = {}
    for name, iso_list in mol_dict.items():
        fname = derive_fname(job_dir, name)
        T_pred = np.loadtxt(fname, skiprows=4)[:, 1]
        T_pred -= T_pred.min()
        trans_dict = {name: transitions[name]}
        for name_iso in iso_list:
            fname = derive_fname(job_dir, name_iso)
            tmp = np.loadtxt(fname, skiprows=4)[:, 1]
            tmp -= tmp.min()
            T_pred += tmp
            trans_dict[name_iso] = transitions[name_iso]
        T_pred += temp_back
        is_accepted = identify_single(T_obs, T_pred, freq, trans_dict, T_thr, tol)

        ret_dict[name] = {
            "iso": iso_list,
            "is_accepted": is_accepted,
            "T_pred": T_pred
        }
    return ret_dict