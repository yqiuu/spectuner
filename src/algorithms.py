import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
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

    incldue_dict = defaultdict(list)
    for normal_dict in normal_dict_list:
        for name, iso_list in normal_dict.items():
            master_name = master_name_dict[name]
            if master_name is not None:
                incldue_dict[master_name].append(deepcopy(iso_list))

    segment_dict = defaultdict(list)
    for idx, normal_dict in enumerate(normal_dict_list):
        for name in normal_dict:
            master_name = master_name_dict[name]
            if master_name is not None:
                segment_dict[master_name].append(idx)

    return mol_list, segment_dict, incldue_dict


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


def filter_moleclues(idn, pm, segments, include_list, T_pred_data, trans_data, params):
    """Select molecules that have emission lines.

    Args:
        idn (Identification): Optimization result.
        pm (ParameterManager): Parameter manager.
        params (array): Parameters.

    Returns:
        mol_dict_new (dict):
        include_list_new (list):
        params_new (array):
    """
    mols_idn = idn.derive_trans_set(segments, T_pred_data, trans_data)
    params_mol, params_den, params_misc = pm.split_params(params, need_reshape=False)
    mol_list_new = []
    params_den_new = []
    idx_iso = 0
    for item in pm.mol_list:
        mols_new = []
        for mol_name in item["molecules"]:
            if mol_name in mols_idn:
                mols_new.append(mol_name)
                params_den_new.append(params_den[idx_iso])
            idx_iso += 1
        item["molecules"] = mols_new
        mol_list_new.append(item)
    params_den_new = np.array(params_den_new)
    params_new = np.concatenate([params_mol, params_den_new, params_misc])

    include_list_new = []
    for mol_list in include_list:
        include_list_new.append([name for name in mol_list if name in mols_idn])
    return mol_list_new, include_list_new, params_new


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


class Identification:
    def __init__(self, obs_data, T_back, prominence,
                 rel_height=.25, n_eval=5, use_dice=False, T_thr=None, frac_fp=1.):
        height = T_back + prominence
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
        self.freq_data = freq_data
        self.T_obs_data = T_obs_data
        self.spans_obs_data = spans_obs_data

        if T_thr is None:
            T_thr = .5*max([T_obs.max() for T_obs in T_obs_data])

        self.T_back = T_back
        self.height = height
        self.prominence = prominence
        self.rel_height = rel_height
        self.n_eval = n_eval
        self.use_dice = use_dice
        self.T_thr = T_thr
        self.frac_fp = frac_fp

    def derive_trans_set(self, segments, T_pred_data, trans_data):
        """Derive the set that includes the name of all transitions."""
        trans_set = set()
        for args in zip(segments, T_pred_data, trans_data):
            trans_set.update(self._derive_trans_set_sub(*args))
        return trans_set

    def compute_scores(self, segments, T_pred_data, trans_data):
        true_pos_dict = {
            "scores": [],
            "freqs": [],
            "names": []
        }
        false_pos_dict = {
            "scores": [],
            "freqs": [],
            "names": []
        }

        def update_data_dict(data, data_new):
            data["scores"].append(data_new["scores"])
            data["freqs"].append(data_new["freqs"])
            data["names"].extend(data_new["names"])

        for i_segment, T_pred, trans_dict in zip(segments, T_pred_data, trans_data):
            true_pos_dict_sub, false_pos_dict_sub = self._compute_scores_sub(i_segment, T_pred, trans_dict)
            update_data_dict(true_pos_dict, true_pos_dict_sub)
            update_data_dict(false_pos_dict, false_pos_dict_sub)
        true_pos_dict["scores"] = np.concatenate(true_pos_dict["scores"])
        true_pos_dict["freqs"] = np.concatenate(true_pos_dict["freqs"])
        false_pos_dict["scores"] = np.concatenate(false_pos_dict["scores"])
        false_pos_dict["freqs"] = np.concatenate(false_pos_dict["freqs"])
        score = np.sum(true_pos_dict["scores"]) + np.sum(false_pos_dict["scores"])
        return {
            "score": score,
            "true_positive": true_pos_dict,
            "false_positive": false_pos_dict,
        }

    def _derive_trans_set_sub(self, i_segment, T_pred, trans_dict):
        """Derive the set that includes the name of all transitions."""
        trans_set = set()

        freq = self.freq_data[i_segment]
        spans_obs = self.spans_obs_data[i_segment]
        spans_pred, _ = derive_peaks(
            freq, T_pred, self.height, self.prominence, self.rel_height
        )
        if len(spans_pred) == 0:
            return trans_set

        _, _, inds_pred = derive_intersections(spans_obs, spans_pred)
        if len(inds_pred) > 0:
            name_list = self._derive_name_list(trans_dict, spans_pred, inds_pred)
            for names in name_list:
                trans_set.update(set(names))
        return trans_set

    def _compute_scores_sub(self, i_segment, T_pred, trans_dict):
        freq = self.freq_data[i_segment]
        T_obs = self.T_obs_data[i_segment]
        spans_obs = self.spans_obs_data[i_segment]

        # Set default returns
        true_pos_dict = {
            "scores": np.zeros(0),
            "freqs": np.zeros(0),
            "names": []
        }
        false_pos_dict = {
            "scores": np.zeros(0),
            "freqs": np.zeros(0),
            "names": []
        }

        spans_pred, _ = derive_peaks(
            freq, T_pred, self.height, self.prominence, self.rel_height
        )
        if len(spans_pred) == 0:
            return true_pos_dict, false_pos_dict

        spans_inter, inds_obs, inds_pred = derive_intersections(spans_obs, spans_pred)
        if len(spans_inter) > 0:
            values_obs, values_pred, factor = derive_true_postive_props(
                freq, T_obs, T_pred, spans_obs, spans_pred,
                spans_inter, inds_pred, inds_obs, self.n_eval
            )
            errors = np.mean(np.abs(values_pred - values_obs), axis=1)
            norm = np.mean(values_obs, axis=1) - self.T_back
            frac = np.minimum(1, norm/(self.T_thr - self.T_back))
            if not self.use_dice:
                frac = 1.
            true_pos_dict["scores"] = frac*np.maximum(0, factor - errors/norm)
            true_pos_dict["freqs"] = np.mean(spans_inter, axis=1)
            true_pos_dict["names"] = self._derive_name_list(trans_dict, spans_pred, inds_pred)

        spans_fp = derive_complementary(spans_pred, inds_pred)
        if len(spans_fp) != 0:
            values_obs_fp, values_pred_fp \
                = derive_false_postive_props(freq, T_obs, T_pred, spans_fp, self.n_eval)
            errors_fp = np.mean(np.maximum(0, values_pred_fp - values_obs_fp), axis=1)
            norm_fp = np.mean(values_pred_fp, axis=1) - self.T_back
            frac = np.minimum(1, norm_fp/(self.T_thr - self.T_back))
            false_pos_dict["scores"] = -self.frac_fp*frac*errors_fp/norm_fp
            false_pos_dict["freqs"] = np.mean(spans_fp, axis=1)
            false_pos_dict["names"] = self._derive_name_list(trans_dict, spans_fp)

        return true_pos_dict, false_pos_dict

    def _derive_name_list(self, trans_dict, spans, inds=None):
        if inds is None:
            spans_uni = spans
            uni_inverse = None
        else:
            inds_uni, uni_inverse = np.unique(inds, return_inverse=True)
            spans_uni = spans[inds_uni]

        name_list = []
        freq_list = []
        for key, freqs in trans_dict.items():
            name_list.extend([key]*len(freqs))
            freq_list.extend(freqs)
        name_list = np.array(name_list)
        freq_list = np.array(freq_list)

        f_left, f_right = spans_uni.T
        inds = np.searchsorted(f_left, freq_list) - 1
        cond_idx = inds >= 0
        inds[~cond_idx] = 0
        cond_right = freq_list <= f_right[inds]
        cond = cond_idx & cond_right

        names_ret = [[] for _ in range(len(spans_uni))]
        for idx, name in zip(inds[cond], name_list[cond]):
            names_ret[idx].append(name)

        if uni_inverse is not None:
            names_ret = np.array(names_ret, dtype=object)[uni_inverse]
            names_ret = list(names_ret)

        return names_ret


class PeakMatchingLoss:
    def __init__(self, obs_data, T_back, prominence, rel_height, n_eval=5):
        height = T_back + prominence
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
        self.freq_data = freq_data
        self.T_obs_data = T_obs_data
        self.spans_obs_data = spans_obs_data

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
            loss += np.sum(np.sum(errors_fp))

        loss = min(0, loss)
        return loss