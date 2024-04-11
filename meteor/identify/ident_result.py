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


def concat_identify_result(res_list):
    if len(res_list) == 0:
        return

    res_list = [res for res in res_list if len(res.df_mol) > 0]
    df_mol = pd.concat([res.df_mol for res in res_list])
    line_table = {}
    line_table_fp = {}
    T_single_dict = {}
    for res in res_list:
        line_table.update(deepcopy(res.line_table))
        line_table_fp.update(deepcopy(res.line_table_fp))
        T_single_dict.update(deepcopy(res.T_single_dict))
    #df_mol.sort_values(["num_tp_i", "score"], ascending=False, inplace=True)
    df_mol.reset_index(drop=True, inplace=True)
    return IdentResult(
        df_mol=df_mol,
        line_table=line_table,
        line_table_fp=line_table_fp,
        T_single_dict=T_single_dict,
        freq_data=res.freq_data,
        T_back=res.T_back,
        is_sep=True
    )


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
            for name in ["loss", "score", "error", "norm"]:
                getattr(self, name)[inds] = np.nan
            for name in ["frac", "id", "name"]:
                for idx in inds:
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
    df_mol: pd.DataFrame
    line_table: LineTable
    line_table_fp: LineTable
    T_single_dict: dict
    freq_data: list
    T_back: float
    is_sep: bool

    def __post_init__(self):
        mol_dict = defaultdict(list)
        if len(self.df_mol) > 0:
            for i_id, name in zip(self.df_mol["id"], self.df_mol["name"]):
                mol_dict[i_id].append(name)
        self._mol_dict = dict(mol_dict)
        df_mol = self.df_mol
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
        n_mol = len(self.df_mol)
        n_master = len(set(self.df_mol["id"]))
        stats_dict.update(n_master=n_master, n_mol=n_mol)

        if self.is_sep:
            n_idn = 0
            n_tot = 0
            recall = 0.
        else:
            n_idn = 0
            for names in self.line_table["name"]:
                if names is not None:
                    n_idn += 1
            n_tot = len(self.line_table["freq"])
            recall = n_idn/n_tot
        stats_dict.update(n_tot=n_tot, n_idn=n_idn, recall=recall)

        return stats_dict

    def extract(self, key):
        def filter_name_list(target_set, name_list):
            inds = []
            for idx, names in enumerate(name_list):
                if names is None:
                    continue
                if not target_set.isdisjoint(set(names)):
                    inds.append(idx)
            return inds

        df_mol_new = deepcopy(self.df_mol[self.df_mol["id"] == key])
        #
        inds = filter_name_list(set((key,)), self.line_table.id)
        line_table_new = self.line_table.extract(inds, is_sparse=True)
        #
        inds = filter_name_list(set((key,)), self.line_table_fp.id)
        line_table_fp_new = self.line_table.extract(inds, is_sparse=False)
        #
        T_single_dict_new = {key: deepcopy(self.T_single_dict[key])}
        return IdentResult(
            df_mol=df_mol_new,
            line_table=line_table_new,
            line_table_fp=line_table_fp_new,
            T_single_dict=T_single_dict_new,
            freq_data=self.freq_data,
            T_back=self.T_back,
            is_sep=True
        )

    def get_T_pred(self, key=None, name=None):
        if key is not None and name is not None:
            return self.T_single_dict[key][name]
        return sum_T_single_data(self.T_single_dict, self.T_back, key)