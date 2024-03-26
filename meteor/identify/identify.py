import pickle
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from .ident_result import concat_identify_result, IdentResult
from ..xclass_wrapper import combine_mol_stores
from ..preprocess import load_preprocess


def identify(config, parent_dir, target):
    T_back = config["sl_model"].get("tBack", 0.)
    obs_data = load_preprocess(config["files"], T_back)
    prominence = config["pm_loss"]["prominence"]
    rel_height =  config["pm_loss"]["rel_height"]
    idn = Identification(obs_data, T_back, prominence, rel_height)

    if target == "single":
        fname_base = config.get("fname_base", None)
    elif target == "combine":
        fname_base =  Path(parent_dir)/"combine"/"combine.pickle"
    else:
        # The function will always terminate in this block
        target = Path(target)
        if target.exists():
            res = identify_file(idn, target, config)
            save_name = target.parent/f"identify_{target.name}"
            pickle.dump(res, open(save_name, "wb"))
            return
        else:
            raise ValueError(f"Unknown target: {target}.")

    dirname = Path(parent_dir)/target
    if fname_base is None:
        res = identify_without_base(idn, dirname, config)
    else:
        res = identify_with_base(idn, dirname, fname_base, config)
    pickle.dump(res, open(dirname/Path("identify.pickle"), "wb"))


def identify_file(idn, fname, config):
    data = pickle.load(open(fname, "rb"))
    res = idn.identify(
        data["mol_store"], config["sl_model"], data["params_best"],
    )
    return res


def identify_without_base(idn, dirname, config):
    res_list = []
    for fname in dirname.glob("*.pickle"):
        if is_exclusive(fname):
            continue
        data = pickle.load(open(fname, "rb"))
        res = idn.identify(
            data["mol_store"], config["sl_model"], data["params_best"],
        )
        if len(res.df_mol) == 0:
            continue
        res = res.extract_sub(data["mol_store"].mol_list[0]["id"])
        res_list.append(res)
    return concat_identify_result(res_list)


def identify_with_base(idn, dirname, fname_base, config):
    config_slm = config["sl_model"]

    data = pickle.load(open(fname_base, "rb"))
    mol_store_base = data["mol_store"]
    params_base = data["params_best"]
    T_single_dict_base = compute_T_single_data(
        mol_store_base, config_slm, params_base, data["freq"]
    )

    res_list = []
    for fname in dirname.glob("*.pickle"):
        if is_exclusive(fname):
            continue
        data = pickle.load(open(fname, "rb"))
        mol_store_combine, params_combine = combine_mol_stores(
            [mol_store_base, data["mol_store"]],
            [params_base, data["params_best"]],
            config["sl_model"]
        )
        T_single_dict = deepcopy(T_single_dict_base)
        T_single_dict.update(compute_T_single_data(
            data["mol_store"], config_slm, data["params_best"], data["freq"]
        ))
        res = idn.identify(
            mol_store_combine, config_slm, params_combine, T_single_dict
        )
        if len(res.df_mol) == 0:
            continue
        try:
            res = res.extract_sub(data["mol_store"].mol_list[0]["id"])
        except KeyError:
            res = None
        if res is not None:
            res_list.append(res)
    return concat_identify_result(res_list)


def is_exclusive(fname):
    name = str(fname.name)
    return name.startswith("identify") \
        or name.startswith("combine") \
        or name.startswith("tmp")


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
                names.append(name)
                fracs.append(compute_peak_norms(spans_pred, freq, T_pred))
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


def derive_true_postive_props(freq, T_obs, T_pred, T_back, spans_obs, spans_pred,
                              spans_inter, inds_pred, inds_obs):
    """Derive properties used to compute errors of true postive samples."""
    errors = np.zeros(len(spans_inter))
    norms = np.zeros(len(spans_inter))
    d_eval = np.ravel(np.diff(spans_inter, axis=1))
    iterator = enumerate(eval_spans(spans_inter, freq, T_obs, T_pred))
    for i_span, (x_eval, T_obs_eval, T_pred_eval) in iterator:
        errors[i_span] = np.trapz(np.abs(T_obs_eval - T_pred_eval), x_eval)
        norms[i_span] = np.trapz(T_obs_eval, x_eval)
    errors /= d_eval
    norms /= d_eval
    norms -= T_back
    f_dice = compute_dice_score(spans_inter, spans_obs[inds_obs], spans_pred[inds_pred])
    return errors, norms, f_dice


def derive_false_postive_props(freq, T_obs, T_pred, T_back, spans_fp):
    """Derive properties used to compute errors of false postive samples."""
    errors = np.zeros(len(spans_fp))
    norms = np.zeros(len(spans_fp))
    d_eval = np.ravel(np.diff(spans_fp, axis=1))
    iterator = enumerate(eval_spans(spans_fp, freq, T_obs, T_pred))
    for i_span, (x_eval, T_obs_eval, T_pred_eval) in iterator:
        errors[i_span] = np.trapz(np.maximum(0, T_pred_eval - T_obs_eval), x_eval)
        norms[i_span] = np.trapz(T_obs_eval, x_eval)
    errors /= d_eval
    norms /= d_eval
    norms -= T_back
    return errors, norms


def compute_peak_norms(spans, freq, spec):
    norms = np.zeros(len(spans))
    for i_span, (x_eval, y_eval) in enumerate(eval_spans(spans, freq, spec)):
        norms[i_span] = np.trapz(y_eval, x_eval)
    norms /= np.ravel(np.diff(spans))
    return norms


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
            df_mol.sort_values(["num_tp_i", "score"], ascending=False, inplace=True)
            df_mol.reset_index(drop=True, inplace=True)

        return IdentResult(
            df_mol, df_sub_dict, true_pos_dict_sparse, false_pos_dict,
            T_single_dict, self.freq_data, self.T_back, is_sep=False
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
            errors, norms, f_dice = derive_true_postive_props(
                freq, T_obs, T_pred, self.T_back, spans_obs, spans_pred,
                spans_inter, inds_pred, inds_obs
            )
            losses = errors - norms*f_dice
            if not self.use_dice:
                f_dice = 1.
            scores = np.maximum(0, f_dice - errors/norms)
            true_pos_dict["freq"] = np.mean(spans_inter, axis=1)
            true_pos_dict["score"] = scores
            true_pos_dict["loss"] = losses
            true_pos_dict["frac"], true_pos_dict["id"], true_pos_dict["name"] \
                = self._compute_fractions(i_segment, T_single_dict, spans_inter)
            true_pos_dict["inds_obs"] = inds_obs

        spans_fp = derive_complementary(spans_pred, inds_pred)
        if len(spans_fp) != 0:
            errors_fp, norms_fp = derive_false_postive_props(
                freq, T_obs, T_pred, self.T_back, spans_fp
            )
            losses = errors_fp
            scores = -self.frac_fp*errors_fp/norms_fp
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
                fracs.append(compute_peak_norms(spans_inter, freq, T_pred))
                ids.append(i_id)
                names.append(name)
        fracs = compute_contributions(fracs, self.T_back)
        ids = np.array(ids)
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
            errors, norms, f_dice = derive_true_postive_props(
                freq, T_obs, T_pred, self.T_back, spans_obs, spans_pred,
                spans_inter, inds_pred, inds_obs
            )
            loss = np.sum(errors - f_dice*norms)
        else:
            loss = 0.

        spans_fp = derive_complementary(spans_pred, inds_pred)
        if len(spans_fp) != 0:
            errors_fp, _ = derive_false_postive_props(
                freq, T_obs, T_pred, self.T_back, spans_fp
            )
            loss += np.sum(errors_fp)

        return loss
