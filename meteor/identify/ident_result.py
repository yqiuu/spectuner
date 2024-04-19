from functools import partial
from collections import defaultdict
from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np
import pandas as pd


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


@dataclass
class LineTable:
    freq: np.ndarray = field(default_factory=partial(np.zeros, 0))
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
            for name in ["loss", "score", "error", "norm"]:
                setattr(line_table_new, name, getattr(self, name)[inds])
            for name in ["frac", "id", "name"]:
                tmp = []
                for idx in inds:
                    tmp.append(getattr(self, name)[idx])
                setattr(line_table_new, name, tmp)
        return line_table_new


@dataclass
class IdentResult:
    mol_data: dict
    line_table: LineTable
    line_table_fp: LineTable
    T_single_dict: dict
    freq_data: list
    T_back: float

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

    def derive_df_mol(self):
        data = []
        for key, sub_dict in self.mol_data.items():
            for name, cols in sub_dict.items():
                data.append({"id": key, "name": name, **cols})
        df = pd.DataFrame.from_dict(data)
        df.sort_values(
            ["id", "num_tp_i", "score"],
            ascending=[True, False, False],
            inplace=True
        )
        return df

    def derive_df_mol_master(self):
        data = []
        for key, sub_dict in self.mol_data.items():
            for cols in sub_dict.values():
                master_name = cols["master_name"]
                break
            cols = {"id": key, "master_name": master_name}
            for prop_name in ["loss", "score", "num_tp", "num_tp_i", "num_fp"]:
                cols[prop_name] = self.get_aggregate_prop(key, prop_name)
            data.append(cols)
        df = pd.DataFrame.from_dict(data)
        df.sort_values("id")
        return df

    def extract(self, key):
        mol_data_new = {key: deepcopy(self.mol_data[key])}
        #
        inds = self.filter_name_list(set((key,)), self.line_table.id)
        line_table_new = self.line_table.extract(inds, is_sparse=True)
        #
        inds = self.filter_name_list(set((key,)), self.line_table_fp.id)
        line_table_fp_new = self.line_table.extract(inds, is_sparse=False)
        #
        T_single_dict_new = {key: deepcopy(self.T_single_dict[key])}
        return IdentResult(
            mol_data=mol_data_new,
            line_table=line_table_new,
            line_table_fp=line_table_fp_new,
            T_single_dict=T_single_dict_new,
            freq_data=self.freq_data,
            T_back=self.T_back,
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
        return sum_T_single_data(self.T_single_dict, self.T_back, key)