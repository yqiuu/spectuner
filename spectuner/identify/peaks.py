from functools import partial
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .ident_result import (
    compute_T_single_data, sum_T_single_data,
    LineTable, IdentResult
)


def filter_moleclues(mol_store, config, params,
                     freq_data, T_pred_data, T_back, prominence, rel_height,
                     frac_cut=.05):
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
    T_single_dict = compute_T_single_data(mol_store, config, params, freq_data)
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
                names.append(name)
                fracs.append(compute_peak_norms(spans_pred, freq, T_single))
        fracs = compute_contributions(fracs, T_back)
        names = np.array(names, dtype=object)
        for cond in fracs.T > frac_cut:
            names_pos.update(set(names[cond]))

    if len(names_pos) == 0:
        return None, None
    return mol_store.select_subset_with_params(names_pos, params, config)


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


def derive_peaks_multi(freq_data, spec_data, height_list, prom_list, rel_height):
    span_data = []
    cond_data = []
    for freq, spec, height, prominence in zip(freq_data, spec_data, height_list, prom_list):
        if spec is None:
            continue
        spans, heights = derive_peaks(freq, spec, height, prominence, rel_height)
        span_data.append(spans)
        cond_data.append(heights)
    span_data = np.vstack(span_data)
    cond_data = np.concatenate(cond_data)
    inds = np.argsort(span_data[:, 0])
    span_data = span_data[inds]
    cond_data = cond_data[inds]
    return span_data, cond_data


def derive_peaks(freq, spec, height, prominence, rel_height):
    spans, is_inter = find_peaks_inters(spec, height, prominence, rel_height)
    spans = np.interp(np.ravel(spans), np.arange(len(freq)), freq).reshape(-1, 2)
    is_inter = np.array(is_inter)
    return spans, is_inter


def derive_peaks_extra(freq, spec, height, prominence, rel_height, return_rel_h=False):
    ret_h_min = 1e-3
    spans_ret = []
    rel_hs_ret = []
    queue = [(spec, 0, rel_height)]
    while len(queue) != 0 :
        spec_it, offset, rel_h = queue.pop(0)
        spans, conds = find_peaks_inters(spec_it, height, prominence, rel_h)
        for sp, is_inter in zip(spans, conds):
            if is_inter and rel_h > ret_h_min:
                idx_left = int(sp[0])
                idx_right = int(sp[1]) + 1
                queue.append((spec_it[idx_left:idx_right], idx_left, .5*rel_h))
            else:
                spans_ret.append((sp[0] + offset, sp[1] + offset))
                rel_hs_ret.append(rel_h)
    spans_ret = np.array(spans_ret)
    rel_hs_ret = np.array(rel_hs_ret)

    if len(spans_ret) == 0:
        if return_rel_h:
            return spans_ret, rel_hs_ret
        return spans_ret

    spans_ret = np.interp(np.ravel(spans_ret), np.arange(len(freq)), freq).reshape(-1, 2)
    inds = np.argsort(spans_ret[:, 0])
    spans_ret = spans_ret[inds]
    if return_rel_h:
        return spans_ret, rel_hs_ret
    return spans_ret


def find_peaks_inters(x, height, prominence, rel_height):
    peaks, _ = signal.find_peaks(x, height=height, prominence=prominence)
    _, _, f_left, f_right = signal.peak_widths(x, peaks, rel_height)
    spans = [[left, right] for left, right in zip(f_left, f_right)]
    spans.sort(key=lambda x: x[0])

    # Find inter intervals
    spans_new = []
    is_inter = []
    for left, right in spans:
        if len(spans_new) == 0 or spans_new[-1][-1] < left:
            spans_new.append([left, right])
            is_inter.append(False)
        else:
            spans_new[-1][-1] = max(spans_new[-1][-1], right)
            is_inter[-1] = True

    return spans_new, is_inter


def derive_peaks_obs_data(obs_data, height_list, prom_list, rel_height, freqs_exclude):
    freq_data = []
    T_obs_data = []
    spans_obs_data = []
    for spec, height, prominence in zip(obs_data, height_list, prom_list):
        freq = spec[:, 0]
        T_obs = spec[:, 1]
        freq_data.append(freq)
        T_obs_data.append(T_obs)
        spans_obs = derive_peaks(freq, T_obs, height, prominence, rel_height)[0]
        spans_obs_data.append(spans_obs)
    if freqs_exclude is None or len(freqs_exclude) == 0:
        return freq_data, T_obs_data, spans_obs_data

    spans_data_new = []
    for spans in spans_obs_data:
        spans_new = spans[match_spans(spans, freqs_exclude)]
        spans_data_new.append(spans_new)
    return freq_data, T_obs_data, spans_data_new


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


def derive_true_postive_props(freq, T_obs, T_pred, T_back, spans_obs, spans_pred,
                              spans_inter, inds_pred, inds_obs):
    """Derive properties used to compute errors of true postive samples."""
    errors = np.zeros(len(spans_inter))
    norms_obs = np.zeros(len(spans_inter))
    norms_pred = np.zeros(len(spans_inter))
    d_eval = np.ravel(np.diff(spans_inter, axis=1))
    iterator = enumerate(eval_spans(spans_inter, freq, T_obs, T_pred))
    for i_span, (x_eval, T_obs_eval, T_pred_eval) in iterator:
        errors[i_span] = np.trapz(np.abs(T_obs_eval - T_pred_eval), x_eval)
        norms_obs[i_span] = np.trapz(T_obs_eval - T_back, x_eval)
        norms_pred[i_span] = np.trapz(T_pred_eval - T_back, x_eval)
    errors /= d_eval
    norms_obs /= d_eval
    norms_pred /= d_eval
    f_dice = compute_dice_score(spans_inter, spans_obs[inds_obs], spans_pred[inds_pred])
    return errors, norms_obs, norms_pred, f_dice


def derive_false_postive_props(freq, T_obs, T_pred, T_back, spans_fp, prominence):
    """Derive properties used to compute errors of false postive samples."""
    errors = np.zeros(len(spans_fp))
    norms = np.zeros(len(spans_fp))
    d_eval = np.ravel(np.diff(spans_fp, axis=1))
    iterator = enumerate(eval_spans(spans_fp, freq, T_obs, T_pred))
    for i_span, (x_eval, T_obs_eval, T_pred_eval) in iterator:
        err_a = np.trapz(np.maximum(0, T_pred_eval - T_obs_eval), x_eval)
        err_b = np.trapz(np.maximum(0, T_pred_eval - T_back - prominence), x_eval)
        errors[i_span] = min(err_a, err_b)
        norms[i_span] = np.trapz(T_pred_eval - T_back, x_eval)
    errors /= d_eval
    norms /= d_eval
    return errors, norms


def compute_peak_norms(spans, freq, spec):
    norms = np.zeros(len(spans))
    for i_span, (x_eval, y_eval) in enumerate(eval_spans(spans, freq, spec)):
        norms[i_span] = np.trapz(y_eval, x_eval)
    norms /= np.ravel(np.diff(spans))
    return norms


def eval_spans(spans, x, y_a, y_b=None):
    inds_left = np.searchsorted(x, spans[:, 0])
    inds_right = np.searchsorted(x, spans[:, 1]) - 1
    for (x_left, x_right), idx_left, idx_right in zip(spans, inds_left, inds_right):
        x_eval = np.concatenate([[x_left], x[idx_left:idx_right], [x_right]])
        y_a_eval = np.interp(x_eval, x, y_a)
        if y_b is None:
            yield x_eval, y_a_eval
        else:
            y_b_eval = np.interp(x_eval, x, y_b)
            yield x_eval, y_a_eval, y_b_eval


def derive_counts(inds):
    counts = np.zeros(max(inds) + 1)
    for idx in inds:
        counts[idx] += 1
    return counts[inds]


def compute_dice_score(spans_inter, spans_a, spans_b):
    return np.ravel(2*np.diff(spans_inter)/(np.diff(spans_a) + np.diff(spans_b)))


def derive_peak_params(prominence, T_back, n_segment):
    if isinstance(prominence, float):
        prom_list = [prominence]*n_segment
    else:
        assert n_segment == len(prominence), \
            "The Number of promiences must be equal to the number of segments."
        prom_list = prominence
    height_list = [T_back + prominence for prominence in prom_list]
    return height_list, prom_list


def match_spans(spans, freqs):
    inds = np.searchsorted(freqs, spans[:, 0])
    cond = inds != len(freqs)
    inds[~cond] = len(freqs) - 1
    cond &= spans[:, 1] >= freqs[inds]
    return ~cond


def create_spans(freqs, v_min, v_max):
    """Create spans using minimum and maximum velocities."""
    freqs_min = compute_shift(freqs, v_min)
    freqs_max = compute_shift(freqs, v_max)
    return np.vstack([freqs_min, freqs_max]).T


def compute_shift(freq, vel):
    c = 3e5 # speed of light [km/s]
    return freq*(1 + vel/c)


@np.vectorize
def linear_deacy(x, x_left, x_right, side, height):
    if side == 1:
        return height*(x - x_left)/(x_right - x_left)
    if side == -1:
        return height*(x_right - x)/(x_right - x_left)

    x_mid = .5*(x_left + x_right)
    if x < x_mid:
        return height*(x - x_left)/(x_mid - x_left)
    else:
        return height*(x_right - x)/(x_right - x_mid)


class PeakManager:
    def __init__(self, obs_data, T_back, prominence, rel_height,
                 T_base_data=None, pfactor=None, frac_cut=.05, freqs_exclude=None):
        height_list, prom_list = derive_peak_params(prominence, T_back, len(obs_data))
        self.freq_data, T_obs_data, self.spans_obs_data \
            = derive_peaks_obs_data(obs_data, height_list, prom_list, rel_height, freqs_exclude)
        if T_base_data is None:
            self.T_obs_data = T_obs_data
        else:
            self.T_obs_data = [T_obs - T_base + T_back for T_obs, T_base
                               in zip(T_obs_data, T_base_data)]
        self.T_back = T_back
        self.height_list = height_list
        self.prom_list = prom_list
        self.rel_height = rel_height
        self.pfactor = pfactor
        self.frac_cut = frac_cut

        if pfactor is None:
            self.scale = None
        else:
            self.scale = self._derive_scale(pfactor)

    def __call__(self, T_pred_data):
        loss_delta = 0.
        for T_obs, T_pred in zip(self.T_obs_data, T_pred_data):
            loss_delta += np.mean(self.transform(np.abs(T_obs - T_pred)))
        loss_delta /= len(T_pred_data)

        loss_ex = 0.
        for i_segment, T_pred in enumerate(T_pred_data):
            peak_store = self.create_peak_store(i_segment, T_pred)
            loss_tp, loss_fp = self.compute_loss(i_segment, peak_store)
            loss_ex += np.sum(loss_tp) + np.sum(loss_fp)

        return loss_delta + loss_ex, loss_ex

    def create_peak_store(self, i_segment, T_pred):
        freq = self.freq_data[i_segment]
        T_obs = self.T_obs_data[i_segment]
        spans_obs = self.spans_obs_data[i_segment]
        spans_pred, _ = derive_peaks(
            freq, T_pred,
            self.height_list[i_segment], self.prom_list[i_segment],
            self.rel_height
        )
        if len(spans_pred) == 0:
            return PeakStore(spans_obs)

        spans_inter, inds_obs, inds_pred = derive_intersections(spans_obs, spans_pred)
        if len(spans_inter) > 0:
            errors_tp, norms_tp_obs, norms_tp_pred, f_dice = derive_true_postive_props(
                freq, T_obs, T_pred, self.T_back, spans_obs, spans_pred,
                spans_inter, inds_pred, inds_obs
            )
        else:
            spans_inter = np.zeros((0, 2))
            errors_tp = np.zeros(0)
            norms_tp_obs = np.zeros(0)
            norms_tp_pred = np.zeros(0)
            f_dice = np.zeros(0)

        spans_fp = derive_complementary(spans_pred, inds_pred)
        if len(spans_fp) > 0:
            errors_fp, norms_fp = derive_false_postive_props(
                freq, T_obs, T_pred, self.T_back, spans_fp, self.prom_list[i_segment],
            )
        else:
            spans_fp = np.zeros((0, 2))
            errors_fp = np.zeros(0)
            norms_fp = np.zeros(0)

        return PeakStore(
            spans_obs, spans_pred, spans_inter, inds_obs, inds_pred,
            errors_tp, norms_tp_obs, norms_tp_pred, f_dice,
            spans_fp, errors_fp, norms_fp
        )

    def compute_loss(self, i_segment, peak_store):
        freq = self.freq_data[i_segment]
        if len(peak_store.spans_inter) > 0:
            loss_tp = self.transform(peak_store.errors_tp) \
                - peak_store.f_dice*self.transform(peak_store.norms_tp_obs)
            cond = peak_store.norms_tp_pred < peak_store.norms_tp_obs
            loss_tp[cond] = np.minimum(loss_tp[cond], 0)
        else:
            loss_tp = np.zeros(0)

        if len(peak_store.spans_fp) > 0:
            centres_obs = np.mean(peak_store.spans_obs, axis=1)
            centres_pred = np.mean(peak_store.spans_fp, axis=1)
            side = np.zeros(len(centres_pred), dtype="i4")
            inds_r = np.searchsorted(centres_obs, centres_pred)
            inds_l = inds_r - 1
            # Right
            cond = inds_r == len(centres_obs)
            inds_r[cond] = len(centres_obs) - 1
            x_right = centres_obs[inds_r]
            x_right[cond] = freq[-1]
            side[cond] = 1
            # Left
            cond = inds_l < 0
            inds_l[cond] = 0
            x_left = centres_obs[inds_l]
            x_left[cond] = freq[0]
            side[cond] = -1
            #
            values = self.transform(peak_store.errors_fp)
            loss_fp = linear_deacy(centres_pred, x_left, x_right, side, values)
        else:
            loss_fp = np.zeros(0)

        return loss_tp, loss_fp

    def transform(self, x):
        if self.scale is None:
            return x
        return self.scale*np.log(1 + x/self.scale)

    def compute_score(self, peak_store):
        if len(peak_store.spans_inter) > 0:
            score_tp = np.maximum(1 - peak_store.errors_tp/peak_store.norms_tp_obs, 0.)
            cond = peak_store.norms_tp_obs <= 0.
            score_tp[cond] = 0.
        else:
            score_tp = np.zeros(0)

        if len(peak_store.spans_fp) > 0:
            score_fp = -peak_store.errors_fp/peak_store.norms_fp
        else:
            score_fp = np.zeros(0)

        return score_tp, score_fp

    def identify(self, mol_store, config, params, T_single_dict=None):
        line_table, line_table_fp, T_single_dict = self.derive_line_table(
            mol_store, config, params, T_single_dict
        )
        param_dict = self.derive_param_dict(mol_store, config, params)
        id_set = set()
        id_set.update(self.derive_mol_set(line_table.id))
        id_set.update(self.derive_mol_set(line_table_fp.id))
        name_set = set()
        name_set.update(self.derive_mol_set(line_table.name))
        name_set.update(self.derive_mol_set(line_table_fp.name))
        mol_data = self.derive_mol_data(mol_store, param_dict, id_set, name_set)
        return IdentResult(
            mol_data, line_table, line_table_fp, T_single_dict,
            self.freq_data, self.T_back
        )

    def derive_mol_data(self, mol_store, param_dict, id_set, mol_set):
        data_tree = defaultdict(dict)
        for item in mol_store.mol_list:
            i_id = item["id"]
            if i_id not in id_set:
                continue
            for mol in item["molecules"]:
                if mol not in mol_set:
                    continue
                cols = {"master_name": item["root"]}
                cols.update(param_dict[i_id][mol])
                data_tree[i_id][mol] = cols
        return dict(data_tree)

    def derive_param_dict(self, mol_store, config, params):
        param_mgr = mol_store.create_parameter_manager(config)
        params_mol = param_mgr.derive_mol_params(params)

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

    def derive_line_table(self, mol_store, config, params, T_single_dict):
        if T_single_dict is None:
            T_single_dict = compute_T_single_data(mol_store, config, params, self.freq_data)
        T_pred_data = sum_T_single_data(T_single_dict, self.T_back)
        line_table = LineTable()
        line_table_fp = LineTable()
        for i_segment, T_pred in enumerate(T_pred_data):
            if T_pred is None:
                continue
            line_table_sub, line_table_fp_sub \
                = self._derive_line_table_sub(i_segment, T_pred, T_single_dict)
            line_table.append(line_table_sub)
            line_table_fp.append(line_table_fp_sub)
        return line_table, line_table_fp, T_single_dict

    def derive_mol_set(self, lines):
        mol_set = set()
        for name in lines:
            if name is None:
                continue
            mol_set.update(set(name))
        return mol_set

    def _derive_scale(self, pfactor):
        norms = []
        for spans, freqs, T_obs in zip(self.spans_obs_data, self.freq_data, self.T_obs_data):
            norms.append(compute_peak_norms(spans, freqs, T_obs - self.T_back))
        norms = np.concatenate(norms)
        return pfactor*np.median(norms)

    def _derive_line_table_sub(self, i_segment, T_pred, T_single_dict):
        peak_store = self.create_peak_store(i_segment, T_pred)
        freqs = np.mean(peak_store.spans_obs, axis=1)
        loss_tp, loss_fp = self.compute_loss(i_segment, peak_store)
        score_tp, score_fp = self.compute_score(peak_store)
        frac_list_tp, id_list_tp, name_list_tp = self._compute_fractions(
            i_segment, T_single_dict, peak_store.spans_inter
        )
        line_table = LineTable()
        line_table_tmp = LineTable(
            freq=freqs,
            span=peak_store.spans_obs,
            loss=loss_tp,
            score=score_tp,
            frac=frac_list_tp,
            id=id_list_tp,
            name=name_list_tp,
            error=peak_store.errors_tp,
            norm=peak_store.norms_tp_obs
        )
        sparsity = (peak_store.inds_inter_obs, len(peak_store.spans_obs))
        line_table.append(line_table_tmp, sparsity=sparsity)

        freq_fp = np.mean(peak_store.spans_fp, axis=1)
        frac_list_fp, id_list_fp, name_list_fp = self._compute_fractions(
            i_segment, T_single_dict, peak_store.spans_fp
        )
        line_table_fp = LineTable(
            freq=freq_fp,
            span=peak_store.spans_fp,
            loss=loss_fp,
            score=score_fp,
            frac=frac_list_fp,
            id=id_list_fp,
            name=name_list_fp,
            error=peak_store.errors_fp,
            norm=peak_store.norms_fp
        )
        return line_table, line_table_fp

    def _compute_fractions(self, i_segment, T_single_dict, spans_inter):
        frac_list = []
        id_list = []
        name_list = []
        if len(spans_inter) == 0:
            return frac_list, id_list, name_list

        fracs = []
        ids = []
        names = []
        freq = self.freq_data[i_segment]
        for i_id, sub_dict in T_single_dict.items():
            for name, T_pred_data in sub_dict.items():
                T_pred = T_pred_data[i_segment]
                if T_pred is None:
                    continue
                fracs.append(compute_peak_norms(spans_inter, freq, T_pred))
                ids.append(i_id)
                names.append(name)
        fracs = compute_contributions(fracs, self.T_back)
        ids = np.array(ids)
        names = np.array(names, dtype=object)

        #
        for i_inter, cond in enumerate(fracs.T > self.frac_cut):
            fracs_sub = fracs[cond, i_inter]
            fracs_sub = fracs_sub/np.sum(fracs_sub)
            frac_list.append(fracs_sub)
            id_list.append(tuple(ids[cond]))
            name_list.append(tuple(names[cond]))
        return frac_list, id_list, name_list


@dataclass(frozen=True)
class PeakStore:
    spans_obs: np.ndarray
    spans_pred: np.ndarray = field(default_factory=partial(np.zeros, (0, 2)))
    spans_inter: np.ndarray = field(default_factory=partial(np.zeros, (0, 2)))
    inds_inter_obs: np.ndarray = field(default_factory=partial(np.zeros, 0))
    inds_inter_pred: np.ndarray = field(default_factory=partial(np.zeros, 0))
    errors_tp: np.ndarray = field(default_factory=partial(np.zeros, 0))
    norms_tp_obs: np.ndarray = field(default_factory=partial(np.zeros, 0))
    norms_tp_pred: np.ndarray = field(default_factory=partial(np.zeros, 0))
    f_dice: np.ndarray = field(default_factory=partial(np.zeros, 0))
    spans_fp: np.ndarray = field(default_factory=partial(np.zeros, (0, 2)))
    errors_fp: np.ndarray = field(default_factory=partial(np.zeros, 0))
    norms_fp: np.ndarray = field(default_factory=partial(np.zeros, 0))