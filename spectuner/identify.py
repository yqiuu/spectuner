import json
from typing import Optional, Literal
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from functools import partial
from dataclasses import dataclass, field, asdict

import h5py
import numpy as np
import pandas as pd
import torch

from .slm_factory import (
    combine_specie_lists, sum_T_single_data, compute_T_single_data,
    SpectralLineModelFactory
)
from .sl_model import SpectralLineDB
from .peaks import compute_peak_norms, compute_shift
from .utils import (
    load_result_list, load_fitting_result, load_result_combine,
    hdf_save_dict, hdf_load_dict, derive_specie_save_name
)


def identify(config, target, mode=None, sl_db=None):
    """Perform identification.

    Args:
        config (dict): Config.
        target (str): Directory that saves fitting results.
        mode (str):
            - ``single``: Use ``fname_base`` given in in the config as base
            data.
            - ``combine``: Use the combined fitting results as base data.
    """
    if config["inference"]["ckpt"] is not None:
        # TODO: Allow to use different parameterization
        ckpt = torch.load(
            config["inference"]["ckpt"],
            map_location="cpu",
            weights_only=True
        )
        config["sl_model"]["params"] = ckpt["config"]["sl_model"]["params"]
    use_f_dice = config["identify"]["use_f_dice"]

    slm_factory = SpectralLineModelFactory(config, sl_db=sl_db)
    idn = Identification(slm_factory, config["obs_info"])

    target = Path(target)
    if target.is_file():
        fname = target
    elif target.is_dir():
        if mode == "single":
            fname = target/"results_single.h5"
        elif mode == "combine":
            fname = target/"results_combine.h5"
        else:
            raise ValueError(f"Unknown mode: {mode}.")
    else:
        raise ValueError(f"Unknown target: {target}.")

    if mode == "single":
        fname_base = config.get("fname_base", None)
        if fname_base is None:
            base_data = None
        else:
            base_data = load_result_combine(fname_base)
    elif mode == "combine":
        base_data = load_result_combine(fname)

    pred_data_list = load_result_list(fname)
    if base_data is None:
        res_dict = identify_without_base(idn, pred_data_list, use_f_dice)
    else:
        res_dict = identify_with_base(idn, pred_data_list, base_data, use_f_dice)
    with h5py.File(fname.parent/f"identify_{fname.name}", "w") as fp:
        if mode == "combine" and base_data is not None:
            res = idn.identify(
                base_data["specie"], base_data["x"], use_f_dice=use_f_dice
            )
            res.save_hdf(fp.create_group("combine"))
        for key, res in res_dict.items():
            res.save_hdf(fp.create_group(key))


def identify_without_base(idn, pred_data_list, use_f_dice):
    res_dict = {}
    for data in pred_data_list:
        res = idn.identify(
            data["specie"], data["x"], use_f_dice=use_f_dice
        )
        if res.is_empty():
            continue
        assert len(data["specie"]) == 1
        res_dict[derive_specie_save_name(data["specie"][0])] = res
    return res_dict


def identify_with_base(idn, pred_data_list, base_data, use_f_dice):
    specie_list_base = base_data["specie"]
    params_base = base_data["x"]
    T_single_dict_base = compute_T_single_data(
        idn._slm_factory, idn._obs_info, specie_list_base, params_base
    )

    res_dict = {}
    for data in pred_data_list:
        specie_list_combine, params_combine = combine_specie_lists(
            [specie_list_base, data["specie"]],
            [params_base, data["x"]],
        )
        T_single_dict = deepcopy(T_single_dict_base)
        T_single_dict.update(compute_T_single_data(
            idn._slm_factory, idn._obs_info, data["specie"], data["x"]
        ))
        res = idn.identify(
            specie_list_combine, params_combine, T_single_dict,
            use_f_dice=use_f_dice
        )
        if res.is_empty():
            continue
        assert len(data["specie"]) == 1
        key = data["specie"][0]["id"]
        try:
            res = res.extract(key)
        except KeyError:
            continue
        res_dict[derive_specie_save_name(data["specie"][0])] = res
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
    def __init__(self,
                 slm_factory: SpectralLineModelFactory,
                 obs_info: list,
                 T_base_data: Optional[list]=None,
                 frac_cut: float=0.05):
        self._slm_factory = slm_factory
        self._obs_info = obs_info
        self._peak_mgr = slm_factory.create_peak_mgr(obs_info, T_base_data)
        self._frac_cut = frac_cut

    def identify(self, specie_list, params,
                 T_single_dict=None, use_f_dice=False):
        line_table, line_table_fp, T_single_dict = self.derive_line_table(
            specie_list, params, T_single_dict, use_f_dice
        )
        param_dict = self.derive_param_dict(specie_list, params)
        id_set = set()
        id_set.update(self.derive_mol_set(line_table.id))
        id_set.update(self.derive_mol_set(line_table_fp.id))
        name_set = set()
        name_set.update(self.derive_mol_set(line_table.name))
        name_set.update(self.derive_mol_set(line_table_fp.name))
        specie_data = self.derive_specie_data(specie_list, param_dict, id_set, name_set)
        return IdentResult(
            specie_data, line_table, line_table_fp, T_single_dict,
            self._peak_mgr.freq_data
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

    def derive_param_dict(self, specie_list: list, params: np.ndarray) -> dict:
        param_mgr = self._slm_factory.create_parameter_mgr(specie_list, self._obs_info)
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

    def derive_line_table(self, specie_list, params, T_single_dict, use_f_dice):
        if T_single_dict is None:
            T_single_dict = compute_T_single_data(
                self._slm_factory, self._obs_info, specie_list, params
            )
        T_pred_data = sum_T_single_data(T_single_dict)
        line_table = LineTable()
        line_table_fp = LineTable()
        for i_segment, T_pred in enumerate(T_pred_data):
            if T_pred is None:
                continue
            line_table_sub, line_table_fp_sub \
                = self._derive_line_table_sub(i_segment, T_pred, T_single_dict, use_f_dice)
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
        peak_mgr = self._peak_mgr
        norms = []
        for spans, freqs, T_obs in \
            zip(peak_mgr.spans_obs_data, peak_mgr.freq_data, peak_mgr.T_obs_data):
            norms.append(compute_peak_norms(spans, freqs, T_obs))
        norms = np.concatenate(norms)
        return pfactor*np.median(norms)

    def _derive_line_table_sub(self, i_segment, T_pred, T_single_dict, use_f_dice):
        peak_mgr = self._peak_mgr
        peak_store = peak_mgr.create_peak_store(i_segment, T_pred)
        freqs = np.mean(peak_store.spans_obs, axis=1)
        loss_tp, loss_fp = peak_mgr.compute_loss(i_segment, peak_store)
        score_tp, score_fp = peak_mgr.compute_score(peak_store, use_f_dice)
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
        freq = self._peak_mgr.freq_data[i_segment]
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
        for i_inter, cond in enumerate(fracs.T > self._frac_cut):
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
    frac: list = field(default_factory=list)
    id: list = field(default_factory=list)
    name: list = field(default_factory=list)
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

    def save_hdf(self, fp):
        save_dict = asdict(self)

        ignore_list = ["frac", "id", "name"]
        hdf_save_dict(fp, save_dict, ignore_list=ignore_list)

        indices = []
        frac_save = []
        id_save = []
        name_save = []
        for idx, (frac_list, id_list, name_list) in \
            enumerate(zip(save_dict["frac"], save_dict["id"], save_dict["name"])):
            if frac_list is None:
                continue
            for frac, id_, name in zip(frac_list, id_list, name_list):
                indices.append(idx)
                frac_save.append(frac)
                id_save.append(id_)
                name_save.append(name)
        save_dict_ex = {
            "index": np.asarray(indices),
            "frac": np.asarray(frac_save),
            "id": np.asarray(id_save),
        }
        hdf_save_dict(fp, save_dict_ex)
        fp.create_dataset("name", data=json.dumps(name_save))

    @classmethod
    def load_hdf(cls, fp):
        load_dict = {}
        hdf_load_dict(fp, load_dict, ignore_list=["index", "frac", "id", "name"])

        indices = np.asarray(fp["index"])
        frac_data = np.asarray(fp["frac"])
        id_data = np.asarray(fp["id"])
        name_data = json.loads(fp["name"][()])

        num = len(load_dict["freq"])
        frac_load = [[] for _ in range(num)]
        id_load = [[] for _ in range(num)]
        name_load = [[] for _ in range(num)]
        for idx, frac, id_, name in zip(indices, frac_data, id_data, name_data):
            frac_load[idx].append(frac)
            id_load[idx].append(id_)
            name_load[idx].append(name)

        load_dict["frac"] = [None if len(frac_list) == 0 else np.asarray(frac_list)
                             for frac_list in frac_load]
        load_dict["id"] = [None if len(id_list) == 0 else tuple(id_list)
                           for id_list in id_load]
        load_dict["name"] = [None if len(name_list) == 0 else tuple(name_list)
                             for name_list in name_load]

        return cls(**load_dict)


@dataclass
class IdentResult:
    specie_data: dict
    line_table: LineTable
    line_table_fp: LineTable
    T_single_dict: dict
    freq_data: list

    def __post_init__(self):
        self._add_score_data()
        self._add_count_data()

    def __repr__(self):
        text = "Molecules:\n"
        for key, sub_dict in self.specie_data.items():
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
        self.update_specie_data(score_dict)

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
        self.update_specie_data(count_dict)

    def is_empty(self):
        return len(self.specie_data) == 0

    def update_specie_data(self, data):
        for key, sub_dict in self.specie_data.items():
            for name, cols in sub_dict.items():
                cols.update(data[key][name])

    def get_aggregate_prop(self, key, prop_name):
        return sum([cols[prop_name] for cols in self.specie_data[key].values()])

    def derive_stats_dict(self):
        stats_dict = {
            "n_master": len(self.specie_data),
            "n_mol": sum([len(sub_dict) for sub_dict in self.specie_data.values()])
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
        for key, sub_dict in self.specie_data.items():
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
        for key, sub_dict in self.specie_data.items():
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
            key_list = self.specie_data.keys()
        else:
            key_list = set()
            for sub_dict in self.specie_data.values():
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
        specie_data_new = {key: deepcopy(self.specie_data[key])}
        #
        inds = self.filter_name_list(set((key,)), self.line_table.id)
        line_table_new = self.line_table.extract(inds, is_sparse=True)
        #
        inds = self.filter_name_list(set((key,)), self.line_table_fp.id)
        line_table_fp_new = self.line_table_fp.extract(inds, is_sparse=False)
        #
        T_single_dict_new = {key: deepcopy(self.T_single_dict[key])}
        return IdentResult(
            specie_data=specie_data_new,
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

    def get_identified_lines(self, include_fp=False):
        freqs = []
        for freq, names in zip(self.line_table.freq, self.line_table.name):
            if names is not None:
                freqs.append(freq)
        freqs = np.asarray(freqs)
        if include_fp:
            freqs = np.sort(np.append(freqs, self.line_table_fp.freq))
        return freqs

    def query_sl_dict(self,
                      sl_db: SpectralLineDB,
                      key: int,
                      name: str) -> dict:
        """Query transitions properties.

        This method provides frequencies corrected using the velocity offset.

        Args:
            sl_db: SpectralLineDB object.
            key: Molecular ID.
            name: Molecular name.

        Returns:
            A dict with the following keys:

            -  freq: Corrected frequency in MHz.
            -  freq_rest: Rest-frame frequency in MHz.
            -  E_low: Enerygy of the lower state in K.
            -  E_up: Enerygy of the upper state in K.
            -  A_ul: Einstein A cofficient in s^-1.
            -  g_u: Upper state degeneracy.
            -  Q_T: Partition function.
            -  x_T: Temperature of the partition function.
        """
        v_offset = self.specie_data[key][name]["v_offset"]
        sl_dict = sl_db.query_sl_dict(name, self.freq_data, v_enlarge=v_offset)
        sl_dict["freq_rest"] = sl_dict["freq"]
        sl_dict["freq"] = compute_shift(sl_dict["freq"], -v_offset)
        sl_dict.pop("segment", None)
        return sl_dict

    def save_hdf(self, fp):
        save_dict = {
            "T_single_dict": self.T_single_dict,
            "freq_data": self.freq_data
        }
        hdf_save_dict(fp, save_dict)
        fp.create_dataset("specie_data", data=json.dumps(self.specie_data))
        self.line_table.save_hdf(fp.create_group("line_table"))
        self.line_table_fp.save_hdf(fp.create_group("line_table_fp"))

    @classmethod
    def load_hdf(cls, fp):
        load_dict = {}
        hdf_load_dict(
            fp, load_dict,
            ignore_list=["specie_data", "line_table", "line_table_fp"]
        )
        load_dict["specie_data"] = {int(key): data for key, data in
                                    json.loads(fp["specie_data"][()]).items()}
        load_dict["line_table"] = LineTable.load_hdf(fp["line_table"])
        load_dict["line_table_fp"] = LineTable.load_hdf(fp["line_table_fp"])

        # Convert the keys of T_single_dict to int
        T_single_dict = {}
        for key, T_single in load_dict["T_single_dict"].items():
            T_single_dict[int(key)] = T_single
        load_dict["T_single_dict"] = T_single_dict
        return cls(**load_dict)


class ResultManager:
    """Interface to load fitting and identification results.

    Args:
        dirname: Directory that stores the results.
    """
    def __init__(self, dirname: str):
        dirname = Path(dirname)
        file_list = (
            ("fitting_single", "results_single.h5"),
            ("fitting_combine", "results_combine.h5"),
            ("fitting_modified", "combine_modified.h5"),
            ("ident_single", "identify_results_single.h5"),
            ("ident_combine", "identify_results_combine.h5"),
            ("ident_modified", "identify_combine_modified.h5")
        )
        for key, fname in file_list:
            fname_ = dirname/fname
            attr_name_1 = "_f_{}".format(key)
            attr_name_2 = "_names_{}".format(key)
            if fname_.is_file():
                setattr(self, attr_name_1, fname_)
                setattr(self, attr_name_2, tuple(h5py.File(fname_).keys()))
            else:
                setattr(self, attr_name_1, None)
                setattr(self, attr_name_2, tuple())

    def __repr__(self):
        text = ""
        text += "Fitting results (single):\n"
        for name in self._names_fitting_single:
            text += "  - {}\n".format(name)
        text += "\n"
        text += "Identification results (single):\n"
        for name in self._names_ident_single:
            text += "  - {}\n".format(name)
        text += "\n"
        text += "Fitting results (combine):\n"
        for name in self._names_fitting_combine:
            text += "  - {}\n".format(name)
        text += "\n"
        text += "Identification results (combine):\n"
        for name in self._names_ident_combine:
            text += "  - {}\n".format(name)
        text += "\n"
        text += "Fitting results (modified):\n"
        for name in self._names_fitting_modified:
            text += "  - {}\n".format(name)
        text += "\n"
        text += "Identification results (modified):\n"
        for name in self._names_ident_modified:
            text += "  - {}\n".format(name)
        text += "\n"
        return text

    def _validate_target(self, target):
        if target not in ["single", "combine", "modified"]:
            raise ValueError("Set target from ['single', 'combine', 'modified'].")

    def list_fitting_results(self,
                             target: Literal["single", "combine", "modified"]) -> tuple:
        """List all fitting results.

        Args:
            target: Category name.

        Returns:
            List of fitting result names.
        """
        self._validate_target(target)
        return getattr(self, f"_names_fitting_{target}")

    def list_ident_results(self,
                           target: Literal["single", "combine", "modified"]) -> tuple:
        """List all identification results.

        Args:
            target: Category name.

        Returns:
            List of identification result names.
        """
        self._validate_target(target)
        return getattr(self, f"_names_ident_{target}")

    def load_fitting_result(self,
                            target: Literal["single", "combine", "modified"],
                            name: str) -> dict:
        """Load a fitting result.

        Args:
            target: Category name.
            name: Name of the fitting result.

        Returns:
            Fitting result.
        """
        self._validate_target(target)
        fname = getattr(self, f"_f_fitting_{target}")
        if fname is None:
            raise ValueError(f"Fail to find any fitting results in '{target}'")
        with h5py.File(fname) as fp:
            res = load_fitting_result(fp[name])
        return res

    def load_ident_result(self,
                          target: Literal["single", "combine", "modified"],
                          name: str) -> IdentResult:
        """Load an identification result.

        Args:
            target: Category name.
            name: Name of the identification result.

        Returns:
            Identification result.
        """
        self._validate_target(target)
        fname = getattr(self, f"_f_ident_{target}")
        if fname is None:
            raise ValueError(f"Fail to find any identification results in '{target}'")
        with h5py.File(fname) as fp:
            res = IdentResult.load_hdf(fp[name])
        return res

    def derive_df_mol_master(self,
                             target: Literal["single", "combine"]="combine",
                             max_order: int=3) -> pd.DataFrame:
        """Derive a dataframe that summarizes the identification results of all
        candidates.

        Args:
            target: Category name.
            max_order: Number of the top-x scores to include.

        Returns:
            Identification result summary.
        """
        self._validate_target(target)
        fname = getattr(self, f"_f_ident_{target}")
        if fname is None:
            raise ValueError(f"Fail to find any identification results in '{target}'")

        res_list = []
        with h5py.File(fname) as fp:
            for key, grp in fp.items():
                if key.startswith("combine"):
                    continue
                res_list.append(IdentResult.load_hdf(grp))
        if len(res_list) == 0:
            return
        df_mol_master = pd.concat([res.derive_df_mol_master(max_order=max_order)
                                   for res in res_list])
        df_mol_master.sort_values(
            ["t3_score", "t2_score", "t1_score", "num_tp_i"],
            ascending=False, inplace=True
        )
        df_mol_master.reset_index(drop=True, inplace=True)
        return df_mol_master