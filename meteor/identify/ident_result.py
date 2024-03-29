from dataclasses import dataclass
from copy import deepcopy

import pandas as pd

from . import identify


def concat_identify_result(res_list):
    if len(res_list) == 0:
        return

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
    df_mol.sort_values(["num_tp_i", "score"], ascending=False, inplace=True)
    df_mol.reset_index(drop=True, inplace=True)
    return IdentResult(
        df_mol=df_mol,
        df_sub_dict=df_sub_dict,
        line_dict=line_dict,
        false_line_dict=false_line_dict,
        T_single_dict=T_single_dict,
        freq_data=res.freq_data,
        T_back=res.T_back,
        is_sep=True
    )


@dataclass
class IdentResult:
    df_mol: pd.DataFrame
    df_sub_dict: dict
    line_dict: dict
    false_line_dict: dict
    T_single_dict: dict
    freq_data: list
    T_back: object
    is_sep: bool

    def __post_init__(self):
        self._mol_dict = {key: tuple(df["name"]) for key, df in self.df_sub_dict.items()}
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
        return IdentResult(
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
        return identify.sum_T_single_data(self.T_single_dict, self.T_back, key)

    def plot_T_pred(self, plot, y_min, y_max, key=None, name=None,
                    color_spec="r", alpha=.8, show_lines=True, T_base_data=None):
        T_data = self.get_T_pred(key, name)
        if T_base_data is not None:
            for i_segment, T_base in enumerate(T_base_data):
                if T_base_data is None or T_data[i_segment] is None:
                    continue
                T_data[i_segment] = T_data[i_segment] + T_base - self.T_back
        plot.plot_spec(self.freq_data, T_data, color=color_spec, alpha=alpha)

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

    def plot_unknown_lines(self, plot, y_min, y_max, color="grey", linestyle="-"):
        freqs = []
        for freq, names in zip(self.line_dict["freq"], self.line_dict["name"]):
            if names is None:
                freqs.append(freq)
        plot.vlines(freqs, y_min, y_max, colors=color, linestyles=linestyle)
