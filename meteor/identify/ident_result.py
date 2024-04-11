from collections import defaultdict
from dataclasses import dataclass
from copy import deepcopy

import pandas as pd

from . import identify


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
class IdentResult:
    df_mol: pd.DataFrame
    line_table: object
    line_table_fp: object
    T_single_dict: dict
    freq_data: list
    T_back: float
    is_sep: bool

    def __post_init__(self):
        mol_dict = defaultdict(list)
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

    def extract_sub(self, key):
        df_mol_new = deepcopy(self.df_mol[self.df_mol["id"] == key])
        #
        inds = self.filter_name_list(set((key,)), self.line_table["id"])
        line_table_ = {key: arr[inds] for key, arr in self.line_table.items()}
        line_table_new = {key: deepcopy(line_table_)}
        #
        inds = self.filter_name_list(set((key,)), self.line_table_fp["id"])
        line_table_fp_ = {key: arr[inds] for key, arr in self.line_table_fp.items()}
        line_table_fp_new = {key: deepcopy(line_table_fp_)}
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
        return identify.sum_T_single_data(self.T_single_dict, self.T_back, key)

    def filter_name_list(self, target_set, name_list):
        inds = []
        for idx, names in enumerate(name_list):
            if names is None:
                continue
            if not target_set.isdisjoint(set(names)):
                inds.append(idx)
        return inds