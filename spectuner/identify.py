import pickle
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from functools import partial
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .sl_model import (
    compute_T_single_data, sum_T_single_data, combine_specie_lists,
    ParameterManager
)
from .peaks import compute_peak_norms, PeakManager
from .utils import load_pred_data
from .preprocess import load_preprocess


def identify(config, target, mode=None):
    """Perform identification.

    Args:
        config (dict): Config.
        target (str):
            - If ``target`` is a file, perfrom identification for the target
              file.
            - If ``target`` is a directory, perform identification for all files
              in the directory.

        mode (str): This is only applicable when ``target`` is a directory.
            - ``single``: Use ``fname_base`` given in in the config as base
            data.
            - ``combine``: Use ``combine.pickle`` in the target directory
            as base data.
    """
    obs_data = load_preprocess(config["obs_info"])
    idn = Identification(obs_data, **config["peak_manager"])

    target = Path(target)
    if target.is_file():
        res = identify_file(idn, target, config)
        save_name = target.parent/f"identify_{target.name}"
        pickle.dump(res, open(save_name, "wb"))
    elif target.is_dir():
        if mode == "single":
            fname_base = config.get("fname_base", None)
        elif mode == "combine":
            fname_base = target/"combine"/"combine.pickle"
        else:
            raise ValueError(f"Unknown mode: {mode}.")

        dirname = target/mode
        if fname_base is None:
            res = identify_without_base(idn, dirname, config)
        else:
            res = identify_with_base(idn, dirname, fname_base, config)
        pickle.dump(res, open(dirname/Path("identify.pickle"), "wb"))
    else:
        raise ValueError(f"Unknown target: {target}.")


def identify_file(idn, fname, config):
    data = pickle.load(open(fname, "rb"))
    res = idn.identify(data["specie"], config, data["params_best"])
    return res


def identify_without_base(idn, dirname, config):
    pred_data_list = load_pred_data(dirname.glob("*.pickle"), reset_id=True)
    res_dict = {}
    for data in pred_data_list:
        assert len(data["specie"]) == 1
        key = data["specie"][0]["id"]
        res = idn.identify(data["specie"], config, data["params_best"])
        if res.is_empty():
            res = None
        res_dict[key] = res
    return res_dict


def identify_with_base(idn, dirname, fname_base, config):
    data = pickle.load(open(fname_base, "rb"))
    specie_list_base = data["specie"]
    params_base = data["params_best"]
    T_single_dict_base = compute_T_single_data(
        specie_list_base, config, params_base, data["freq"]
    )

    pred_data_list = load_pred_data(dirname.glob("*.pickle"), reset_id=False)
    res_dict = {}
    for data in pred_data_list:
        specie_list_combine, params_combine = combine_specie_lists(
            [specie_list_base, data["specie"]],
            [params_base, data["params_best"]],
        )
        T_single_dict = deepcopy(T_single_dict_base)
        T_single_dict.update(compute_T_single_data(
            data["specie"], config, data["params_best"], data["freq"]
        ))
        res = idn.identify(
            specie_list_combine, config, params_combine, T_single_dict
        )
        if res.is_empty():
            continue
        assert len(data["specie"]) == 1
        key = data["specie"][0]["id"]
        try:
            res = res.extract(key)
        except KeyError:
            res = None
        res_dict[key] = res
    return res_dict


def compute_contributions(values, T_back=0.):
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


class Identification:
    def __init__(self, obs_data, prominence, rel_height,
                 T_base_data=None, pfactor=None, freqs_exclude=None, frac_cut=.05):
        self.peak_mgr = PeakManager(
            obs_data, prominence, rel_height,
            T_base_data=T_base_data,
            pfactor=pfactor,
            freqs_exclude=freqs_exclude
        )
        self.frac_cut = frac_cut

    def identify(self, specie_list, config, params, T_single_dict=None):
        line_table, line_table_fp, T_single_dict = self.derive_line_table(
            specie_list, config, params, T_single_dict
        )
        param_dict = self.derive_param_dict(specie_list, config, params)
        id_set = set()
        id_set.update(self.derive_mol_set(line_table.id))
        id_set.update(self.derive_mol_set(line_table_fp.id))
        name_set = set()
        name_set.update(self.derive_mol_set(line_table.name))
        name_set.update(self.derive_mol_set(line_table_fp.name))
        specie_data = self.derive_specie_data(specie_list, param_dict, id_set, name_set)
        return IdentResult(
            specie_data, line_table, line_table_fp, T_single_dict,
            self.peak_mgr.freq_data
        )

    def derive_specie_data(self, specie_list, param_dict, id_set, mol_set):
        data_tree = defaultdict(dict)
        for item in specie_list:
            i_id = item["id"]
            if i_id not in id_set:
                continue
            for mol in item["species"]:
                if mol not in mol_set:
                    continue
                cols = {"master_name": item["root"]}
                cols.update(param_dict[i_id][mol])
                data_tree[i_id][mol] = cols
        return dict(data_tree)

    def derive_param_dict(self, specie_list, config, params):
        param_mgr = ParameterManager.from_config(specie_list, config)
        params_mol = param_mgr.derive_params(params)
        param_names = param_mgr.param_names
        param_dict = defaultdict(dict)
        idx = 0
        for item in specie_list:
            for name in item["species"]:
                param_dict[item["id"]][name] \
                    = {key: par for key, par in zip(param_names, params_mol[idx])}
                idx += 1
        param_dict = dict(param_dict)
        return param_dict

    def derive_line_table(self, specie_list, config, params, T_single_dict):
        if T_single_dict is None:
            T_single_dict = compute_T_single_data(
                specie_list, config, params, self.peak_mgr.freq_data
            )
        T_pred_data = sum_T_single_data(T_single_dict)
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
        peak_mgr = self.peak_mgr
        norms = []
        for spans, freqs, T_obs in \
            zip(peak_mgr.spans_obs_data, peak_mgr.freq_data, peak_mgr.T_obs_data):
            norms.append(compute_peak_norms(spans, freqs, T_obs))
        norms = np.concatenate(norms)
        return pfactor*np.median(norms)

    def _derive_line_table_sub(self, i_segment, T_pred, T_single_dict):
        peak_mgr = self.peak_mgr
        peak_store = peak_mgr.create_peak_store(i_segment, T_pred)
        freqs = np.mean(peak_store.spans_obs, axis=1)
        loss_tp, loss_fp = peak_mgr.compute_loss(i_segment, peak_store)
        score_tp, score_fp = peak_mgr.compute_score(peak_store)
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
        freq = self.peak_mgr.freq_data[i_segment]
        for i_id, sub_dict in T_single_dict.items():
            for name, T_pred_data in sub_dict.items():
                T_pred = T_pred_data[i_segment]
                if T_pred is None:
                    continue
                fracs.append(compute_peak_norms(spans_inter, freq, T_pred))
                ids.append(i_id)
                names.append(name)
        fracs = compute_contributions(fracs)
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


@dataclass
class LineTable:
    freq: np.ndarray = field(default_factory=partial(np.zeros, 0))
    span: np.ndarray = field(default_factory=partial(np.zeros, (0, 2)))
    loss: np.ndarray = field(default_factory=partial(np.zeros, 0))
    score: np.ndarray = field(default_factory=partial(np.zeros, 0))
    frac: np.ndarray = field(default_factory=list)
    id: np.ndarray = field(default_factory=list)
    name: np.ndarray = field(default_factory=list)
    error: np.ndarray = field(default_factory=partial(np.zeros, 0))
    norm: np.ndarray = field(default_factory=partial(np.zeros, 0))

    def __len__(self):
        return len(self.freq)

    def append(self, line_table, sparsity=None):
        self.freq = np.append(self.freq, line_table.freq)
        self.span = np.vstack([self.span, line_table.span])

        for name in ["loss", "score", "error", "norm"]:
            if sparsity is None:
                arr_new = getattr(line_table, name)
            else:
                inds, num = sparsity
                arr_tmp = getattr(line_table, name)
                arr_new = np.full(num, np.nan)
                if len(inds) > 0:
                    arr_new[inds] = arr_tmp
            setattr(self, name, np.append(getattr(self, name), arr_new))

        for name in ["frac", "id", "name"]:
            if sparsity is None:
                list_new = getattr(line_table, name)
            else:
                inds, num = sparsity
                list_new = [None for _ in range(num)]
                for idx, val in zip(inds, getattr(line_table, name)):
                    list_new[idx] = val
            getattr(self, name).extend(list_new)

    def extract(self, inds, is_sparse):
        if is_sparse:
            line_table_new = deepcopy(self)
            inds_c = [idx for idx in range(len(self)) if idx not in inds]
            for name in ["loss", "score", "error", "norm"]:
                getattr(self, name)[inds_c] = np.nan
            for name in ["frac", "id", "name"]:
                for idx in inds_c:
                    getattr(line_table_new, name)[idx] = None
        else:
            line_table_new = LineTable()
            for name in ["freq", "span", "loss", "score", "error", "norm"]:
                setattr(line_table_new, name, getattr(self, name)[inds])
            for name in ["frac", "id", "name"]:
                tmp = []
                for idx in inds:
                    tmp.append(getattr(self, name)[idx])
                setattr(line_table_new, name, tmp)
        return line_table_new

    def save_line_table(self, fname):
        fp = open(fname, "w")
        fp.write("# Line ID\n")
        fp.write("# Frequency [MHz]\n")
        fp.write("# Identified Species\n")
        idx = 0
        for freq, name_list in zip(self.freq, self.name):
            if name_list is None:
                continue
            for name in name_list:
                fp.write("{},{:.2f},{}\n".format(idx, freq, name))
            idx += 1
        fp.close()


@dataclass
class IdentResult:
    mol_data: dict
    line_table: LineTable
    line_table_fp: LineTable
    T_single_dict: dict
    freq_data: list

    def __post_init__(self):
        self._add_score_data()
        self._add_count_data()

    def __repr__(self):
        text = "Molecules:\n"
        for key, sub_dict in self.mol_data.items():
            for cols in sub_dict.values():
                master_name = cols["master_name"]
                break
            text += "id={}, {}\n".format(key, master_name)
            for name in sub_dict:
                text += " - {}\n".format(name)
        return text

    def _add_score_data(self):
        def increase_score_dict(line_table, score_dict):
            scores = line_table.score
            losses = line_table.loss
            frac_list = line_table.frac
            id_list = line_table.id
            name_list = line_table.name
            for i_line in range(len(frac_list)):
                if id_list[i_line] is None:
                    continue
                for i_blen in range(len(frac_list[i_line])):
                    key = id_list[i_line][i_blen]
                    name = name_list[i_line][i_blen]
                    frac = frac_list[i_line][i_blen]
                    loss = losses[i_line]*frac
                    score = scores[i_line]*frac
                    score_dict[key][name]["loss"] += loss
                    score_dict[key][name]["score"] += score

        dict_factory = lambda: {"loss": 0., "score": 0.}
        score_dict = defaultdict(lambda: defaultdict(dict_factory))
        increase_score_dict(self.line_table, score_dict)
        increase_score_dict(self.line_table_fp, score_dict)
        self.update_mol_data(score_dict)

    def _add_count_data(self):
        def increase_count_dict(line_table, count_dict, target):
            for id_list, name_list in zip(line_table.id, line_table.name):
                if id_list is None:
                    continue
                for key, name in zip(id_list, name_list):
                    count_dict[key][name][target] += 1

        def increase_count_i_dict(line_table, count_dict, target):
            for id_list, name_list in zip(line_table.id, line_table.name):
                if id_list is None:
                    continue
                if len(id_list) == 1:
                    count_dict[id_list[0]][name_list[0]][target] += 1

        dict_factory = lambda: {"num_tp": 0, "num_tp_i": 0, "num_fp": 0}
        count_dict = defaultdict(lambda: defaultdict(dict_factory))
        increase_count_dict(self.line_table, count_dict, "num_tp")
        increase_count_dict(self.line_table_fp, count_dict, "num_fp")
        increase_count_i_dict(self.line_table, count_dict, "num_tp_i")
        self.update_mol_data(count_dict)

    def is_empty(self):
        return len(self.mol_data) == 0

    def update_mol_data(self, data):
        for key, sub_dict in self.mol_data.items():
            for name, cols in sub_dict.items():
                cols.update(data[key][name])

    def get_aggregate_prop(self, key, prop_name):
        return sum([cols[prop_name] for cols in self.mol_data[key].values()])

    def derive_stats_dict(self):
        stats_dict = {
            "n_master": len(self.mol_data),
            "n_mol": sum([len(sub_dict) for sub_dict in self.mol_data.values()])
        }

        n_idn = 0
        for names in self.line_table.name:
            if names is not None:
                n_idn += 1
        n_tot = len(self.line_table.freq)
        recall = n_idn/n_tot
        stats_dict.update(n_tot=n_tot, n_idn=n_idn, recall=recall)

        return stats_dict

    def derive_df_mol(self, max_order=3):
        tx_score_dict = self.compute_tx_score(max_order, use_id=False)
        data = []
        for key, sub_dict in self.mol_data.items():
            for name, cols in sub_dict.items():
                data.append({"id": key, "name": name, **cols, **tx_score_dict[name]})
        df = pd.DataFrame.from_dict(data)
        df.sort_values(
            ["id", "num_tp_i", "score"],
            ascending=[True, False, False],
            inplace=True
        )
        return df

    def derive_df_mol_master(self, max_order=3):
        tx_score_dict = self.compute_tx_score(max_order, use_id=True)
        data = []
        for key, sub_dict in self.mol_data.items():
            for cols in sub_dict.values():
                master_name = cols["master_name"]
                break
            cols = {"id": key, "master_name": master_name}
            for prop_name in ["loss", "score", "num_tp", "num_tp_i", "num_fp"]:
                cols[prop_name] = self.get_aggregate_prop(key, prop_name)
            if key in tx_score_dict:
                cols.update(tx_score_dict[key])
            data.append(cols)
        df = pd.DataFrame.from_dict(data)
        df.sort_values("id")
        return df

    def compute_tx_score(self, max_order, use_id):
        def compute(score_list, order):
            if len(score_list) < order:
                return 0.
            return score_list[order - 1]

        score_list_dict = defaultdict(list)
        if use_id:
            line_table_key = self.line_table.id
        else:
            line_table_key = self.line_table.name
        iterator = zip(line_table_key, self.line_table.score, self.line_table.frac)
        for id_list, score, frac in iterator:
            if id_list is None:
                continue
            for key, score_sub in zip(id_list, score*frac):
                score_list_dict[key].append(score_sub)

        if use_id:
            key_list = self.mol_data.keys()
        else:
            key_list = set()
            for sub_dict in self.mol_data.values():
                key_list.update(sub_dict.keys())

        score_dict = {}
        for key in key_list:
            if key in score_list_dict:
                score_list = score_list_dict[key]
                score_list.sort(reverse=True)
                score_dict[key] = {f"t{order}_score": compute(score_list, order)
                                   for order in range(1, max_order + 1)}
            else:
                score_dict[key] = {f"t{order}_score": 0.
                                   for order in range(1, max_order + 1)}
        return score_dict

    def extract(self, key):
        mol_data_new = {key: deepcopy(self.mol_data[key])}
        #
        inds = self.filter_name_list(set((key,)), self.line_table.id)
        line_table_new = self.line_table.extract(inds, is_sparse=True)
        #
        inds = self.filter_name_list(set((key,)), self.line_table_fp.id)
        line_table_fp_new = self.line_table_fp.extract(inds, is_sparse=False)
        #
        T_single_dict_new = {key: deepcopy(self.T_single_dict[key])}
        return IdentResult(
            mol_data=mol_data_new,
            line_table=line_table_new,
            line_table_fp=line_table_fp_new,
            T_single_dict=T_single_dict_new,
            freq_data=self.freq_data,
        )

    def filter_name_list(self, target_set, name_list):
        inds = []
        for idx, names in enumerate(name_list):
            if names is None:
                continue
            if not target_set.isdisjoint(set(names)):
                inds.append(idx)
        return inds

    def get_T_pred(self, key=None, name=None):
        if key is not None and name is not None:
            return self.T_single_dict[key][name]
        return sum_T_single_data(self.T_single_dict, key=key)

    def get_unknown_lines(self):
        freqs = []
        for freq, names in zip(self.line_table.freq, self.line_table.name):
            if names is None:
                freqs.append(freq)
        freqs = np.asarray(freqs)
        return freqs

    def get_identified_lines(self):
        freqs = []
        for freq, names in zip(self.line_table.freq, self.line_table.name):
            if names is not None:
                freqs.append(freq)
        freqs = np.asarray(freqs)
        return freqs