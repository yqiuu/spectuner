import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from .atoms import MolecularDecomposer
from .xclass_wrapper import task_ListDatabase, extract_line_frequency


def select_molecules(FreqMin, FreqMax, ElowMin, ElowMax,
                     molecules, elements, base_only,
                     iso_list=None, exclude_list=None):
    if exclude_list is None:
        exclude_list = []
    if molecules is None:
        molecules = []
        skip = True
    else:
        skip = False
    if iso_list is None:
        iso_list = []

    normal_dict = group_by_normal_form(
        FreqMin, FreqMax, ElowMin, ElowMax, elements, exclude_list
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


def select_molecules_multi(obs_data, ElowMin, ElowMax,
                           molecules, elements, base_only,
                           iso_list=None, exclude_list=None):
    if exclude_list is None:
        exclude_list = []
    if molecules is None:
        molecules = []
        skip = True
    else:
        skip = False
    if iso_list is None:
        iso_list = []

    normal_dict_list = []
    for spec in obs_data:
        normal_dict_list.append(group_by_normal_form(
            FreqMin=spec[0, 0],
            FreqMax=spec[-1, 0],
            ElowMin=ElowMin,
            ElowMax=ElowMax,
            elements=elements,
            exclude_list=exclude_list
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

    mol_dict, master_name_dict \
        = replace_with_master_name(normal_dict_all, molecules, base_only, iso_list)
    if not skip:
        mol_dict = {name: mol_dict[name] for name in molecules if name in mol_dict}

    incldue_dict = defaultdict(list)
    for normal_dict in normal_dict_list:
        for name, mol_list in normal_dict.items():
            master_name = master_name_dict[name]
            if master_name is not None:
                incldue_dict[master_name].append(deepcopy(mol_list))

    segment_dict = defaultdict(list)
    for idx, normal_dict in enumerate(normal_dict_list):
        for name in normal_dict:
            master_name = master_name_dict[name]
            if master_name is not None:
                segment_dict[master_name].append(idx)

    if skip:
        return mol_dict, segment_dict, incldue_dict

    mol_dict_ret = {}
    segment_dict_ret = {}
    incldue_dict_ret = {}
    for name in molecules:
        if name in mol_dict:
            mol_dict_ret[name] = mol_dict[name]
            segment_dict_ret[name] = segment_dict[name]
            incldue_dict_ret[name] = incldue_dict[name]
    return mol_dict_ret, segment_dict_ret, incldue_dict_ret


def group_by_normal_form(FreqMin, FreqMax, ElowMin, ElowMax, elements, exclude_list):
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

    # Filter elements
    elements = set(elements)
    normal_dict = defaultdict(list)
    for name in mol_names:
        fm_normal, atom_set = derive_normal_form(name)
        if len(atom_set - elements) == 0:
            normal_dict[fm_normal].append(name)
    return normal_dict


def derive_normal_form(mol_name):
    fm, *_ = mol_name.split(";")
    atom_dict = MolecularDecomposer(fm).ShatterFormula()
    atom_set = set(atom_dict.keys())

    pattern = r"-([0-9])([0-9])[-]?"
    fm = re.sub(pattern, "", fm)
    for pattern in re.findall("[A-Z][a-z]?\d", fm):
        fm = fm.replace(pattern, pattern[:-1]*int(pattern[-1]))
    fm = fm.replace("D", "H")
    return fm, atom_set


def replace_with_master_name(normal_dict, molecules, base_only, iso_list):
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
                and name not in molecules and name not in iso_list:
                continue
            if name == master_name:
                mol_dict[master_name]
            else:
                mol_dict[master_name].append(name)
    return mol_dict, master_name_dict


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


def identify_single_v2(name, obs_data, T_pred_data, trans_data,
                       n_match=2, tol=.15, frac_cut=.25,
                       idx_limit=7, lower_frac=.5):
    T_obs = np.concatenate([spec[:, 1] for spec in obs_data])
    T_max = T_obs.max()
    T_median = np.median(T_obs)
    T_thr = T_median + frac_cut*(T_max - T_median)

    span_data = []
    T_c_data = []
    name_list = []
    segment_inds = []

    for i_segment, (spec, T_pred, trans_dict) \
        in enumerate(zip(obs_data, T_pred_data, trans_data)):
        span_data_sub, T_c_data_sub, name_list_sub \
            = find_peak_span(T_pred, spec[:, 0], trans_dict, idx_limit, lower_frac)
        span_data.extend(span_data_sub)
        T_c_data.extend(T_c_data_sub)
        name_list.extend(name_list_sub)
        segment_inds.extend([i_segment]*len(span_data_sub))

    span_data = np.array(span_data, dtype=object)
    T_c_data = np.array(T_c_data)
    name_list = np.array(name_list)
    segment_inds = np.array(segment_inds)

    cond = T_c_data > T_thr
    span_data = span_data[cond]
    T_c_data = T_c_data[cond]
    name_list = name_list[cond]
    segment_inds = segment_inds[cond]

    errors = []
    errors_neg = []
    freq_c_data = []
    for i_segment, span in zip(segment_inds, span_data):
        T_pred = T_pred_data[i_segment][span]
        T_obs = obs_data[i_segment][span, 1]
        errors.append(np.mean(np.abs(T_obs - T_pred))/np.mean(T_obs))
        errors_neg.append(np.mean(T_obs - T_pred)/np.mean(T_obs))
        freq_c_data.append(obs_data[i_segment][span, 0][np.argmax(T_pred)])
    errors = np.array(errors)
    errors_neg = np.array(errors_neg)
    freq_c_data = np.array(freq_c_data)
    n_match_ = np.count_nonzero(errors < tol)

    if len(span_data) == 0:
        status = "reject"
    elif n_match_ >= n_match:
        status = "accept"
    elif n_match_ >= 1 and np.count_nonzero(errors_neg < -tol) == 0:
        status = "confuse"
    else:
        status = "reject"
    return IdentifyResult(name, status, n_match_, name_list, freq_c_data, errors, errors_neg)


def identify_single_v3(name, obs_data, T_pred_data, trans_data, T_thr,
                       idx_limit=7, lower_frac=.5, upper_cut=1., error_factor=1.):
    span_data = []
    T_c_data = []
    name_list = []
    segment_inds = []
    for i_segment, (spec, T_pred, trans_dict) \
        in enumerate(zip(obs_data, T_pred_data, trans_data)):
        span_data_sub, T_c_data_sub, name_list_sub \
            = find_peak_span(T_pred, spec[:, 0], trans_dict, idx_limit, lower_frac)
        span_data.extend(span_data_sub)
        T_c_data.extend(T_c_data_sub)
        name_list.extend(name_list_sub)
        segment_inds.extend([i_segment]*len(span_data_sub))
    span_data = np.array(span_data, dtype=object)
    T_c_data = np.array(T_c_data)
    name_list = np.array(name_list)
    segment_inds = np.array(segment_inds)

    errors = []
    freq_c_data = []
    for i_segment, span in zip(segment_inds, span_data):
        T_pred = T_pred_data[i_segment][span]
        T_back = T_pred_data[i_segment].min()
        T_obs = np.maximum(obs_data[i_segment][span, 1], T_back)
        norm = np.mean(T_pred - T_back)
        if norm == 0.:
            err = 1
        else:
            err = np.mean(np.abs(T_obs - T_pred))/norm
        errors.append(err)
        freq_c_data.append(obs_data[i_segment][span, 0][np.argmax(T_pred)])
    errors = np.array(errors)
    freq_c_data = np.array(freq_c_data, dtype=object)

    cond = T_c_data > T_thr
    errors = errors[cond]
    name_list = name_list[cond]
    freq_c_data = freq_c_data[cond]
    score = np.count_nonzero(cond) - error_factor*np.sum(np.minimum(upper_cut, errors))
    return IdentifyResult(
        name, "", score, name_list, freq_c_data, errors, None
    )


def find_peak_span(T_data, freq, trans_dict, idx_limit=7, lower_frac=.5):
    freq_min = freq[0]
    freq_max = freq[-1]
    freq_data = []
    name_list = []
    for name, nu_list in trans_dict.items():
        for nu in nu_list:
            if nu > freq_min and nu < freq_max:
                freq_data.append(nu)
                name_list.append(name)

    span_data = []
    T_c_data = []
    name_list_ret = []
    for nu, name in zip(freq_data, name_list):
        idx_c = np.argmin(np.abs(freq - nu))
        if idx_c != 0 and T_data[idx_c - 1] > T_data[idx_c]:
            idx_c = idx_c - 1
        elif idx_c != len(T_data) - 1 and T_data[idx_c + 1] > T_data[idx_c]:
            idx_c = idx_c + 1

        T_c = T_data[idx_c]
        T_limit = lower_frac*T_c

        if idx_c == 0:
            continue
        idx_b = idx_c
        T_prev = T_c
        i_loop = 0
        while idx_b > 0 and idx_c - idx_b < idx_limit \
            and (T_data[idx_b - 1] < T_prev or np.isclose(T_data[idx_b - 1], T_prev)) \
            and T_data[idx_b - 1] > T_limit:
            idx_b -= 1
            T_prev = T_data[idx_b]
            i_loop += 1
        if i_loop == 0:
            continue

        if idx_c == len(freq) - 1:
            continue
        idx_e = idx_c
        T_prev = T_c
        i_loop = 0
        while idx_e < len(freq) - 1 and idx_e - idx_c < idx_limit \
            and (T_data[idx_e + 1] < T_prev or np.isclose(T_data[idx_e + 1], T_prev)) \
            and T_data[idx_e + 1] > T_limit:
            idx_e += 1
            T_prev = T_data[idx_e]
            i_loop += 1
        if i_loop == 0:
            continue

        span = slice(idx_b, idx_e + 1)
        if span not in span_data:
            span_data.append(span)
            T_c_data.append(T_c)
            name_list_ret.append(name)

    return span_data, T_c_data, name_list_ret


def derive_median_T_obs(obs_data):
    T_obs = np.concatenate([spec[:, 1] for spec in obs_data])
    T_median = np.median(T_obs)
    return T_median


@dataclass
class IdentifyResult:
    name: str
    status: str
    n_match: int
    name_list: np.ndarray
    freq_c_data: np.ndarray
    errors: np.ndarray
    errors_neg: np.ndarray

    def __repr__(self):
        return "name: {}\nstatus: {}\nn_match: {}".format(
            self.name, self.status, self.n_match)