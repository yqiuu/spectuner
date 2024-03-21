import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import signal

from .atoms import MolecularDecomposer
from .xclass_wrapper import (
    task_ListDatabase,
    extract_line_frequency,
)


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


def select_molecules_multi(obs_data, ElowMin, ElowMax,
                           elements, molecules, base_only,
                           iso_list=None, exclude_list=None, rename_dict=None):
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

    incldue_dict = defaultdict(lambda: [[] for _ in range(len(obs_data))])
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


def identify_single_v3(name, obs_data, T_pred_data, trans_data,
                       T_thr=None, median_frac=.25,
                       idx_limit=7, lower_frac=.5, upper_cut=1., f_1=1., f_2=0.):
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
    errors_2 = []
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

        err = np.mean(np.abs(np.minimum(0, T_obs - T_pred)))/norm
        errors_2.append(err)
        freq_c_data.append(obs_data[i_segment][span, 0][np.argmax(T_pred)])
    errors = np.array(errors)
    errors_2 = np.array(errors_2)
    freq_c_data = np.array(freq_c_data, dtype=object)

    if T_thr is None:
        T_thr = derive_median_frac_threshold(obs_data, median_frac)
    cond = T_c_data > T_thr
    errors = errors[cond]
    errors_2 = errors_2[cond]
    name_list = name_list[cond]
    freq_c_data = freq_c_data[cond]
    score = np.count_nonzero(cond) - f_1*np.sum(np.minimum(upper_cut, errors)) \
        - f_2*np.sum(np.minimum(upper_cut, errors_2))
    return IdentifyResult(
        name, "", score, name_list, freq_c_data, errors, errors_2
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

    peaks = []
    for nu, name in zip(freq_data, name_list):
        idx_c = np.argmin(np.abs(freq - nu))
        if idx_c != 0 and T_data[idx_c - 1] > T_data[idx_c]:
            idx_c = idx_c - 1
        elif idx_c != len(T_data) - 1 and T_data[idx_c + 1] > T_data[idx_c]:
            idx_c = idx_c + 1

        T_c = T_data[idx_c]
        T_limit = lower_frac*T_c

        if idx_c == 0 or idx_c == len(freq) - 1:
            continue

        idx_b = idx_c
        T_prev = T_c
        is_left_low = False
        while idx_b > 0 and idx_c - idx_b < idx_limit:
            if T_data[idx_b - 1] < T_prev:
                is_left_low = True
            if (T_data[idx_b - 1] < T_prev or T_data[idx_b - 1] == T_prev) \
                and T_data[idx_b - 1] > T_limit:
                idx_b -= 1
                T_prev = T_data[idx_b]
            else:
                break
        if not is_left_low:
            continue

        idx_e = idx_c
        T_prev = T_c
        is_right_low = False
        while idx_e < len(freq) - 1 and idx_e - idx_c < idx_limit:
            if T_data[idx_e + 1] < T_prev:
                is_right_low = True
            if (T_data[idx_e + 1] < T_prev or np.isclose(T_data[idx_e + 1], T_prev)) \
                and T_data[idx_e + 1] > T_limit:
                idx_e += 1
                T_prev = T_data[idx_e]
            else:
                break
        if not is_right_low:
            continue

        span = slice(idx_b, idx_e + 1)
        peaks.append((span, T_c, name))

    # Merge peaks
    peaks.sort(key=lambda x: x[0].start)
    peaks_new = []
    for item in peaks:
        if len(peaks_new) == 0 or (peaks_new[-1][0].stop - 1 <= item[0].start and
                                   peaks_new[-1][0] != item[0]):
            peaks_new.append(item)
        else:
            span = peaks_new[-1][0]
            span_new = slice(span.start, max(span.stop, item[0].stop))
            item_new = (span_new, *peaks_new[-1][1:])
            peaks_new[-1] = item_new

    if len(peaks_new) == 0:
        span_data = []
        T_c_data = []
        name_list = []
    else:
        span_data, T_c_data, name_list = tuple(zip(*peaks_new))
    return span_data, T_c_data, name_list


def derive_median_frac_threshold(obs_data, median_frac):
    T_obs = np.concatenate([spec[:, 1] for spec in obs_data])
    T_max = T_obs.max()
    T_median = np.median(T_obs)
    T_thr = T_median + median_frac*(T_max - T_median)
    return T_thr


def filter_moleclues(mol_store, config_slm, params,
                     freq_data, T_pred_data, T_back, prominence, rel_height,
                     n_eval=5, frac_cut=.05):
    """Select molecules that have emission lines.

    Args:
        idn (Identification): Optimization result.
        pm (ParameterManager): Parameter manager.
        params (array): Parameters.

    Returns:
        mol_store (MoleculeStore): None if no emission lines.
        params (array): None if no emission lines.
    """
    height = T_back + prominence
    T_single_dict = compute_T_single_data(mol_store, config_slm, params, freq_data)
    names_pos = set()

    for i_segment in range(len(T_pred_data)):
        freq = freq_data[i_segment]
        T_pred = T_pred_data[i_segment]
        if T_pred is None:
            continue
        spans_pred = derive_peaks(freq, T_pred, height, prominence, rel_height)[0]

        names = []
        fracs = []
        for sub_dict in T_single_dict.values():
            for name, T_single_data in sub_dict.items():
                T_single = T_single_data[i_segment]
                if T_single is None:
                    continue
                values = np.mean(eval_spans(spans_pred, freq, T_single, n_eval), axis=1)
                names.append(name)
                fracs.append(values)
        fracs = compute_contributions(fracs, T_back)
        names = np.array(names, dtype=object)
        for cond in fracs.T > frac_cut:
            names_pos.update(set(names[cond]))

    if len(names_pos) == 0:
        return None, None
    return mol_store.select_subset_with_params(names_pos, params, config_slm)


def select_mol_store(mol_store, config_slm, params, names_pos):
    mol_store_sub = mol_store.select_subset(names_pos)
    pm = mol_store.create_parameter_manager(config_slm)
    params_sub = pm.get_subset_params(names_pos, params)
    return mol_store_sub, params_sub


def derive_intersections(spans_a, spans_b):
    inds_a = []
    inds_b = []
    spans_ret = []

    i_a = 0
    i_b = 0
    while i_a < len(spans_a) and i_b < len(spans_b):
        start = max(spans_a[i_a][0], spans_b[i_b][0])
        end = min(spans_a[i_a][1], spans_b[i_b][1])
        if start <= end:
            spans_ret.append((start, end))
            inds_a.append(i_a)
            inds_b.append(i_b)

        if spans_a[i_a][1] < spans_b[i_b][1]:
            i_a += 1
        else:
            i_b += 1

    inds_a = np.array(inds_a)
    inds_b = np.array(inds_b)
    spans_ret = np.array(spans_ret)

    return spans_ret, inds_a, inds_b


def derive_complementary(spans, inds):
    """Derive spans that are not in the given indices."""
    inds_iso = [idx for idx in range(len(spans)) if idx not in set(inds)]
    return spans[inds_iso]


def derive_peaks(freq, spec, height, prominence, rel_height):
    peaks, _ = signal.find_peaks(spec, height=height, prominence=prominence)
    _, peak_heights, f_left, f_right = signal.peak_widths(spec, peaks, rel_height)
    spans = [[left, right] for left, right in zip(f_left, f_right)]
    spans.sort(key=lambda x: x[0])

    # Merge
    spans_new = []
    peak_heights_new = []
    for (left, right), h in zip(spans, peak_heights):
        if len(spans_new) == 0 or spans_new[-1][-1] < left:
            spans_new.append([left, right])
            peak_heights_new.append(h)
        else:
            spans_new[-1][-1] = max(spans_new[-1][-1], right)
            peak_heights_new[-1] = min(peak_heights_new[-1], h)

    spans_new = np.interp(np.ravel(spans_new), np.arange(len(freq)), freq).reshape(-1, 2)
    peak_heights_new = np.array(peak_heights_new)
    return spans_new, peak_heights_new


def derive_peaks_multi(freq_data, spec_data, height, prominence, rel_height):
    spans_data = []
    heights_data = []
    for freq, spec in zip(freq_data, spec_data):
        if spec is None:
            continue
        spans, heights = derive_peaks(freq, spec, height, prominence, rel_height)
        spans_data.append(spans)
        heights_data.append(heights)
    spans_data = np.vstack(spans_data)
    heights_data = np.concatenate(heights_data)
    inds = np.argsort(spans_data[:, 0])
    spans_data = spans_data[inds]
    heights_data = heights_data[inds]
    return spans_data, heights_data


def derive_peaks_obs_data(obs_data, height, prominence, rel_height):
    freq_data = []
    T_obs_data = []
    spans_obs_data = []
    for spec in obs_data:
        freq = spec[:, 0]
        T_obs = spec[:, 1]
        freq_data.append(freq)
        T_obs_data.append(T_obs)
        spans_obs = derive_peaks(freq, T_obs, height, prominence, rel_height)[0]
        spans_obs_data.append(spans_obs)
    return freq_data, T_obs_data, spans_obs_data


def derive_peaks_pred_data(mol_store, config_slm, params,
                           freq_data, T_pred_data, T_back, prominence, rel_height):
    height = T_back + prominence
    spans_tot = derive_peaks_multi(freq_data, T_pred_data, height, prominence, rel_height)[0]
    name_list = [[] for _ in range(len(spans_tot))]
    names_pos = []

    pm = mol_store.create_parameter_manager(config_slm)
    for item in mol_store.mol_list:
        for mol in item["molecules"]:
            params_single = pm.get_single_params(mol, params)
            mol_store_single = mol_store.select_single(mol)
            sl_model = mol_store_single.create_spectral_line_model(config_slm)
            iterator = sl_model.call_multi(
                freq_data, mol_store_single.include_list, params_single, remove_dir=True
            )
            T_pred_single = [args[0] for args in iterator]
            spans_single = derive_peaks_multi(
                freq_data, T_pred_single, height, prominence, rel_height)[0]
            inds = derive_intersections(spans_tot, spans_single)[1]
            for idx in inds:
                name_list[idx].append(mol)
            if len(inds) > 0:
                names_pos.append(mol)

    return spans_tot, name_list, names_pos


def derive_blending_list(obs_data, pred_data_list, T_back, prominence, rel_height):
    height = T_back + prominence
    spans_obs_data = derive_peaks_obs_data(obs_data, height, prominence, rel_height)[-1]
    spans_obs = np.vstack(spans_obs_data)
    spans_obs = spans_obs[np.argsort(spans_obs[:, 0])]

    blending_list = []
    for i_pred, pred_data in enumerate(pred_data_list):
        spans_pred = derive_peaks_multi(
            pred_data["freq"], pred_data["T_pred"], height, prominence, rel_height)[0]
        inds_obs = set(derive_intersections(spans_obs, spans_pred)[1])
        blending_list.append((i_pred, inds_obs))
    blending_list.sort(key=lambda item: -len(item[1]))

    return blending_list


def compute_contributions(values, T_back):
    """Compute the contributions of each blending peaks.

    Args:
        fracs (list): Each element should be an array that gives the mean
            temperature of the peaks. Different elements give the values from
            different molecules.
        T_back (_type_): Background temperature.

    Returns:
        array (N_mol, N_peak): Normalized fractions.
    """
    fracs = np.vstack(values)
    if len(fracs.shape) == 1:
        fracs = fracs[:, None]
    fracs -= T_back
    norm = np.sum(fracs, axis=0)
    norm[norm == 0.] = len(fracs)
    fracs /= norm
    return fracs


def derive_true_postive_props(freq, T_obs, T_pred, spans_obs, spans_pred,
                              spans_inter, inds_pred, inds_obs, n_eval):
    """Derive properties used to compute errors of true postive samples."""
    values_obs = eval_spans(spans_inter, freq, T_obs, n_eval)
    values_pred = eval_spans(spans_inter, freq, T_pred, n_eval)
    f_dice = compute_dice_score(spans_inter, spans_obs[inds_obs], spans_pred[inds_pred])
    return values_obs, values_pred, f_dice


def derive_false_postive_props(freq, T_obs, T_pred, spans_fp, n_eval):
    """Derive properties used to compute errors of false postive samples."""
    values_obs = eval_spans(spans_fp, freq, T_obs, n_eval)
    values_pred = eval_spans(spans_fp, freq, T_pred, n_eval)
    return values_obs, values_pred


def compute_T_single_data(mol_store, config_slm, params, freq_data):
    pm = mol_store.create_parameter_manager(config_slm)
    T_single_data = defaultdict(dict)
    for item in mol_store.mol_list:
        for mol in item["molecules"]:
            params_single = pm.get_single_params(mol, params)
            mol_store_single = mol_store.select_single(mol)
            T_single_data[item["id"]][mol] \
                = mol_store_single.compute_T_pred_data(params_single, freq_data, config_slm)
    T_single_data = dict(T_single_data)
    return T_single_data


def sum_T_single_data(T_single_dict, T_back, key=None):
    # Get a test dict
    for sub_dict in T_single_dict.values():
        for T_single_data in sub_dict.values():
            break
        break
    T_ret_data = [None for _ in T_single_data]

    def sum_sub(target_dict):
        for T_single_data in target_dict.values():
            for i_segment, T_single in enumerate(T_single_data):
                if T_single is None:
                    continue
                if T_ret_data[i_segment] is None:
                    T_ret_data[i_segment] = T_back
                T_ret_data[i_segment] = T_ret_data[i_segment] + T_single - T_back

    if key is not None:
        sum_sub(T_single_dict[key])
        return T_ret_data

    for sub_dict in T_single_dict.values():
        sum_sub(sub_dict)
    return T_ret_data


def quad_simps(x, y, spans):
    x_p = np.hstack([spans, np.mean(spans, axis=1, keepdims=True)])
    y_p = np.interp(np.ravel(x_p), x, y).reshape(*x_p.shape)
    weights = np.array([1./6, 1./6, 2./3])*np.diff(spans)
    return np.sum(weights*y_p, axis=1)


def eval_spans(spans, x, y, n_eval):
    frac = np.linspace(0., 1, n_eval)
    x_p = spans[:, :1] + np.diff(spans)*frac
    y_p = np.interp(np.ravel(x_p), x, y).reshape(*x_p.shape)
    return y_p


def derive_counts(inds):
    counts = np.zeros(max(inds) + 1)
    for idx in inds:
        counts[idx] += 1
    return counts[inds]


def compute_dice_score(spans_inter, spans_a, spans_b):
    return np.ravel(2*np.diff(spans_inter)/(np.diff(spans_a) + np.diff(spans_b)))


def concat_identify_result(res_list):
    res_list = [res for res in res_list if len(res.df_mol) > 0]
    df_mol = pd.concat([res.df_mol for res in res_list])
    df_sub_dict = {}
    line_dict = {}
    false_line_dict = {}
    T_single_dict = {}
    for res in res_list:
        df_sub_dict.update(deepcopy(res.df_sub_dict))
        line_dict.update(deepcopy(res.line_dict))
        false_line_dict.update(deepcopy(res.false_line_dict))
        T_single_dict.update(deepcopy(res.T_single_dict))
    return IdentifyResult(
        df_mol=df_mol,
        df_sub_dict=df_sub_dict,
        line_dict=line_dict,
        false_line_dict=false_line_dict,
        T_single_dict=T_single_dict,
        freq_data=res.freq_data,
        T_back=res.T_back,
        is_sep=True
    )


class Identification:
    def __init__(self, obs_data, T_back, prominence,
                 rel_height=.25, n_eval=5, use_dice=False, is_loss=False,
                 frac_fp=1., frac_cut=.05):
        height = T_back + prominence
        self.freq_data, self.T_obs_data, self.spans_obs_data \
            = derive_peaks_obs_data(obs_data, height, prominence, rel_height)

        self.T_back = T_back
        self.height = height
        self.prominence = prominence
        self.rel_height = rel_height
        self.n_eval = n_eval
        self.use_dice = use_dice
        self.is_loss = is_loss
        self.frac_fp = frac_fp
        self.frac_cut = frac_cut

    def identify(self, mol_store, config_slm, params, T_single_dict=None):
        true_pos_dict, false_pos_dict, true_pos_dict_sparse, T_single_dict \
            = self.compute_scores(mol_store, config_slm, params, T_single_dict)
        score_dict, score_sub_dict \
            = self.derive_score_dict(true_pos_dict, false_pos_dict)
        param_dict = self.derive_param_dict(mol_store, config_slm, params)

        sub_dict = defaultdict(list)
        for item in mol_store.mol_list:
            i_id = item["id"]
            if i_id not in score_sub_dict:
                continue
            for mol in item["molecules"]:
                if mol not in score_sub_dict[i_id]:
                    continue
                cols = {"master_name": item["root"], "name": mol}
                cols.update(score_sub_dict[i_id][mol])
                cols.update(param_dict[i_id][mol])
                sub_dict[i_id].append(cols)
        df_sub_dict = {i_id: pd.DataFrame.from_dict(res_list)
                       for i_id, res_list in sub_dict.items()}

        res_list = []
        for item in mol_store.mol_list:
            i_id = item["id"]
            if i_id not in score_dict:
                continue
            master_name = item["root"]
            cols = {"id": i_id, "master_name": master_name}
            cols.update(score_dict[i_id])
            for p_dict in param_dict[i_id].values():
                cols.update(p_dict)
                break
            cols["den"] = df_sub_dict[i_id]["den"].sum()
            res_list.append(cols)
        df_mol = pd.DataFrame.from_dict(res_list)
        if len(df_mol) > 0:
            df_mol.sort_values("score", ascending=False, inplace=True)

        return IdentifyResult(
            df_mol, df_sub_dict, true_pos_dict_sparse, false_pos_dict,
            T_single_dict, self.freq_data, self.T_back
        )

    def derive_param_dict(self, mol_store, config_slm, params):
        pm = mol_store.create_parameter_manager(config_slm)
        params_ = pm.scaler.derive_params(pm, params)
        params_mol = pm.derive_mol_params(params_)

        param_names = ["size", "T_ex", "den", "delta_v", "v_lsr"]
        param_dict = defaultdict(dict)
        idx = 0
        for item in mol_store.mol_list:
            for mol in item["molecules"]:
                param_dict[item["id"]][mol] \
                    = {key: par for key, par in zip(param_names, params_mol[idx])}
                idx += 1
        param_dict = dict(param_dict)
        return param_dict

    def derive_score_dict(self, true_pos_dict, false_pos_dict):
        def increase_score_dict(score_dict, score_sub_dict, data_dict):
            scores = data_dict["score"]
            losses = data_dict["loss"]
            frac_list = data_dict["frac"]
            id_list = data_dict["id"]
            name_list = data_dict["name"]
            for i_line in range(len(frac_list)):
                for i_blen in range(len(frac_list[i_line])):
                    i_id = id_list[i_line][i_blen]
                    name = name_list[i_line][i_blen]
                    frac = frac_list[i_line][i_blen]
                    loss = losses[i_line]*frac
                    score = scores[i_line]*frac
                    score_dict[i_id]["loss"] += loss
                    score_dict[i_id]["score"] += score
                    score_sub_dict[i_id][name]["loss"] += loss
                    score_sub_dict[i_id][name]["score"] += score

        def increase_number_dict(score_dict, data_dict, key):
            for i_id in score_dict:
                for id_list in data_dict["id"]:
                    if i_id in id_list:
                        score_dict[i_id][key] += 1

        def increase_number_i_dict(score_dict, data_dict, key):
            for i_id in score_dict:
                for id_list in data_dict["id"]:
                    if len(id_list) == 1 and i_id == id_list[0]:
                        score_dict[i_id][key] += 1

        dict_factory = lambda: {"loss": 0., "score": 0., "num_tp": 0, "num_tp_i": 0, "num_fp": 0}
        score_dict = defaultdict(dict_factory)
        dict_factory = lambda: {"loss": 0., "score": 0.}
        score_sub_dict = defaultdict(lambda: defaultdict(dict_factory))
        increase_score_dict(score_dict, score_sub_dict, true_pos_dict)
        increase_score_dict(score_dict, score_sub_dict, false_pos_dict)

        increase_number_dict(score_dict, true_pos_dict, "num_tp")
        increase_number_dict(score_dict, false_pos_dict, "num_fp")
        increase_number_i_dict(score_dict, true_pos_dict, "num_tp_i")
        # Convert to standard dict
        score_dict = dict(score_dict)
        score_sub_dict = {key: dict(val) for key, val in score_sub_dict.items()}
        return score_dict, score_sub_dict

    def compute_scores(self, mol_store, config_slm, params, T_single_dict):
        true_pos_dict = {
            "freq": [],
            "loss": [],
            "score": [],
            "frac": [],
            "id": [],
            "name": []
        }
        false_pos_dict = {
            "freq": [],
            "loss": [],
            "score": [],
            "frac": [],
            "id": [],
            "name": []
        }
        true_pos_dict_sparse = {
            "freq": [],
            "loss": [],
            "score": [],
            "frac": [],
            "id": [],
            "name": []
        }

        def update_data_dict(data, data_new):
            data["freq"].append(data_new["freq"])
            data["loss"].append(data_new["loss"])
            data["score"].append(data_new["score"])
            data["frac"].extend(data_new["frac"])
            data["id"].extend(data_new["id"])
            data["name"].extend(data_new["name"])

        def update_sparse_dict(data, data_new, freq):
            inds_obs = data_new["inds_obs"]
            num = len(freq)
            data["freq"].append(freq)
            for key in ["loss", "score"]:
                tmp = np.zeros(num)
                tmp[inds_obs] = data_new[key]
                data[key].append(tmp)
            for key in ["frac", "id", "name"]:
                tmp = [None for _ in range(num)]
                for idx, val in zip(inds_obs, data_new[key]):
                    tmp[idx] = val
                data[key].extend(tmp)

        if T_single_dict is None:
            T_single_dict = compute_T_single_data(mol_store, config_slm, params, self.freq_data)
        T_pred_data = sum_T_single_data(T_single_dict, self.T_back)
        for i_segment, T_pred in enumerate(T_pred_data):
            if T_pred is None:
                continue
            true_pos_dict_sub, false_pos_dict_sub \
                = self._compute_scores_sub(i_segment, T_pred, T_single_dict)
            update_data_dict(true_pos_dict, true_pos_dict_sub)
            update_data_dict(false_pos_dict, false_pos_dict_sub)
            update_sparse_dict(
                true_pos_dict_sparse, true_pos_dict_sub,
                np.mean(self.spans_obs_data[i_segment], axis=1)
            )
        for key in ["freq", "loss", "score"]:
            true_pos_dict[key] = np.concatenate(true_pos_dict[key])
            false_pos_dict[key] = np.concatenate(false_pos_dict[key])
        for key in ["freq", "loss", "score"]:
            true_pos_dict_sparse[key] = np.concatenate(true_pos_dict_sparse[key])
        for key in ["frac", "id", "name"]:
            true_pos_dict_sparse[key] = np.array(true_pos_dict_sparse[key], dtype=object)
            false_pos_dict[key] = np.array(false_pos_dict[key], dtype=object)
        return true_pos_dict, false_pos_dict, true_pos_dict_sparse, T_single_dict

    def _compute_scores_sub(self, i_segment, T_pred, T_single_dict):
        freq = self.freq_data[i_segment]
        T_obs = self.T_obs_data[i_segment]
        spans_obs = self.spans_obs_data[i_segment]

        # Set default returns
        true_pos_dict = {
            "freq": np.zeros(0),
            "score": np.zeros(0),
            "loss": np.zeros(0),
            "frac": [],
            "id": [],
            "name": [],
            "inds_obs": np.zeros(0, dtype=int)
        }
        false_pos_dict = {
            "freq": np.zeros(0),
            "score": np.zeros(0),
            "loss": np.zeros(0),
            "frac": [],
            "id": [],
            "name": [],
        }

        spans_pred, _ = derive_peaks(
            freq, T_pred, self.height, self.prominence, self.rel_height
        )
        if len(spans_pred) == 0:
            return true_pos_dict, false_pos_dict

        spans_inter, inds_obs, inds_pred = derive_intersections(spans_obs, spans_pred)
        if len(spans_inter) > 0:
            values_obs, values_pred, f_dice = derive_true_postive_props(
                freq, T_obs, T_pred, spans_obs, spans_pred,
                spans_inter, inds_pred, inds_obs, self.n_eval
            )
            errors = np.mean(np.abs(values_pred - values_obs), axis=1)
            norm = np.mean(values_obs, axis=1) - self.T_back
            losses = errors - norm*f_dice
            if not self.use_dice:
                f_dice = 1.
            scores = np.maximum(0, f_dice - errors/norm)
            true_pos_dict["freq"] = np.mean(spans_inter, axis=1)
            true_pos_dict["score"] = scores
            true_pos_dict["loss"] = losses
            true_pos_dict["frac"], true_pos_dict["id"], true_pos_dict["name"] \
                = self._compute_fractions(i_segment, T_single_dict, spans_inter)
            true_pos_dict["inds_obs"] = inds_obs

        spans_fp = derive_complementary(spans_pred, inds_pred)
        if len(spans_fp) != 0:
            values_obs_fp, values_pred_fp \
                = derive_false_postive_props(freq, T_obs, T_pred, spans_fp, self.n_eval)
            errors_fp = np.mean(np.maximum(0, values_pred_fp - values_obs_fp), axis=1)
            norm_fp = np.mean(values_pred_fp, axis=1) - self.T_back
            losses = errors_fp
            scores = -self.frac_fp*errors_fp/norm_fp
            false_pos_dict["freq"] = np.mean(spans_fp, axis=1)
            false_pos_dict["score"] = scores
            false_pos_dict["loss"] = losses
            false_pos_dict["frac"], false_pos_dict["id"], false_pos_dict["name"] \
                = self._compute_fractions(i_segment, T_single_dict, spans_fp)
        return true_pos_dict, false_pos_dict

    def _compute_fractions(self, i_segment, T_single_dict, spans_inter):
        fracs = []
        ids = []
        names = []
        freq = self.freq_data[i_segment]
        for i_id, sub_dict in T_single_dict.items():
            for name, T_pred_data in sub_dict.items():
                T_pred = T_pred_data[i_segment]
                if T_pred is None:
                    continue
                values = np.mean(eval_spans(spans_inter, freq, T_pred, self.n_eval), axis=1)
                fracs.append(values)
                ids.append(i_id)
                names.append(name)
        fracs = np.vstack(fracs)
        fracs -= self.T_back
        norm = np.sum(fracs, axis=0)
        norm[norm == 0.] = len(fracs)
        fracs /= norm
        ids= np.array(ids)
        names = np.array(names, dtype=object)

        frac_list = []
        id_list = []
        name_list = []
        for i_inter, cond in enumerate(fracs.T > self.frac_cut):
            fracs_sub = fracs[cond, i_inter]
            fracs_sub = fracs_sub/np.sum(fracs_sub)
            frac_list.append(fracs_sub)
            id_list.append(tuple(ids[cond]))
            name_list.append(tuple(names[cond]))
        return frac_list, id_list, name_list


class IdentifyResult:
    def __init__(self, df_mol, df_sub_dict, line_dict, false_line_dict,
                 T_single_dict, freq_data, T_back, is_sep=False):
        self.df_mol = df_mol
        self.df_sub_dict = df_sub_dict
        self.line_dict = line_dict
        self.false_line_dict = false_line_dict
        self.T_single_dict = T_single_dict
        self.freq_data = freq_data
        self.T_back = T_back
        self.is_sep = is_sep

        self._mol_dict = {key: tuple(df["name"]) for key, df in df_sub_dict.items()}
        if len(df_mol) > 0:
            self._master_name_dict = {key: name for key, name
                                      in zip(df_mol["id"], df_mol["master_name"])}
        else:
            self._master_name_dict = {}

    def __repr__(self):
        text = "Molecules:\n"
        for key, name_list in self._mol_dict.items():
            text += "id={}, {}\n".format(key, self._master_name_dict[key])
            for name in name_list:
                text += " - {}\n".format(name)
        return text

    @property
    def mol_dict(self):
        return self._mol_dict

    def derive_stats_dict(self):
        stats_dict = {}
        n_mol = 0
        for df in self.df_sub_dict.values():
            n_mol += len(df)
        n_master = len(self.df_mol)
        stats_dict.update(n_master=n_master, n_mol=n_mol)

        if self.is_sep:
            n_idn = 0
            n_tot = 0
            recall = 0.
        else:
            n_idn = 0
            for names in self.line_dict["name"]:
                if names is not None:
                    n_idn += 1
            n_tot = len(self.line_dict["freq"])
            recall = n_idn/n_tot
        stats_dict.update(n_tot=n_tot, n_idn=n_idn, recall=recall)

        return stats_dict

    def extract_sub(self, key):
        df_mol_new = deepcopy(self.df_mol[self.df_mol["id"] == key])
        df_sub_dict_new = {key: deepcopy(self.df_sub_dict[key])}
        #
        inds = self.filter_name_list(set((key,)), self.line_dict["id"])
        line_dict_ = {key: arr[inds] for key, arr in self.line_dict.items()}
        line_dict_new = {key: deepcopy(line_dict_)}
        #
        inds = self.filter_name_list(set((key,)), self.false_line_dict["id"])
        false_line_dict_ = {key: arr[inds] for key, arr in self.false_line_dict.items()}
        false_line_dict_new = {key: deepcopy(false_line_dict_)}
        #
        T_single_dict_new = {key: deepcopy(self.T_single_dict[key])}
        return IdentifyResult(
            df_mol=df_mol_new,
            df_sub_dict=df_sub_dict_new,
            line_dict=line_dict_new,
            false_line_dict=false_line_dict_new,
            T_single_dict=T_single_dict_new,
            freq_data=self.freq_data,
            T_back=self.T_back,
            is_sep=True
        )

    def get_T_pred(self, key=None, name=None):
        if key is not None and name is not None:
            return self.T_single_dict[key][name]
        return sum_T_single_data(self.T_single_dict, self.T_back, key)

    def plot_T_pred(self, plot, y_min, y_max, key=None, name=None,
                    color_spec="r", alpha=.8, show_lines=True):
        plot.plot_spec(
            self.freq_data, self.get_T_pred(key, name),
            color=color_spec, alpha=alpha
        )
        if not show_lines:
            return

        if key is None:
            plot.plot_names(
                self.line_dict["freq"], self.line_dict["name"],
                y_min, y_max
            )
            plot.plot_names(
                self.false_line_dict["freq"], self.false_line_dict["name"],
                y_min, y_max, color="b"
            )
            return

        if name is None:
            name_set = set(self.T_single_dict[key])
        else:
            name_set = set((name,))
        if self.is_sep:
            line_dict = self.line_dict[key]
            false_line_dict = self.false_line_dict[key]
        else:
            line_dict = self.line_dict
            false_line_dict = self.false_line_dict
        inds = self.filter_name_list(name_set, line_dict["name"])
        spans = line_dict["freq"][inds]
        name_list = line_dict["name"][inds]
        plot.plot_names(spans, name_list, y_min, y_max)
        inds = self.filter_name_list(name_set, false_line_dict["name"])
        spans = false_line_dict["freq"][inds]
        name_list = false_line_dict["name"][inds]
        plot.plot_names(spans, name_list, y_min, y_max, color="b")

    def filter_name_list(self, target_set, name_list):
        inds = []
        for idx, names in enumerate(name_list):
            if names is None:
                continue
            if not target_set.isdisjoint(set(names)):
                inds.append(idx)
        return inds


class PeakMatchingLoss:
    def __init__(self, obs_data, T_back, prominence, rel_height, n_eval=5):
        height = T_back + prominence
        self.freq_data, self.T_obs_data, self.spans_obs_data \
            = derive_peaks_obs_data(obs_data, height, prominence, rel_height)

        self.T_back = T_back
        self.height = height
        self.prominence = prominence
        self.rel_height = rel_height
        self.n_eval = n_eval


    def __call__(self, i_segment, T_pred):
        freq = self.freq_data[i_segment]
        T_obs = self.T_obs_data[i_segment]
        spans_obs = self.spans_obs_data[i_segment]

        spans_pred, _ = derive_peaks(
            freq, T_pred, self.height, self.prominence, self.rel_height
        )
        if len(spans_pred) == 0:
            return 0.

        spans_inter, inds_obs, inds_pred = derive_intersections(spans_obs, spans_pred)
        if len(spans_inter) > 0:
            values_obs, values_pred, factor = derive_true_postive_props(
                freq, T_obs, T_pred, spans_obs, spans_pred,
                spans_inter, inds_pred, inds_obs, self.n_eval
            )
            errors = np.mean(np.abs(values_pred - values_obs), axis=1)
            norm = np.mean(values_obs, axis=1) - self.T_back
            loss = np.sum(errors - factor*norm)
        else:
            loss = 0.

        spans_fp = derive_complementary(spans_pred, inds_pred)
        if len(spans_fp) != 0:
            values_obs_fp, values_pred_fp \
                = derive_false_postive_props(freq, T_obs, T_pred, spans_fp, self.n_eval)
            errors_fp = np.mean(np.maximum(0, values_pred_fp - values_obs_fp), axis=1)
            loss += np.sum(errors_fp)

        return loss