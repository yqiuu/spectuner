import sqlite3
from collections import defaultdict

import numpy as np


class SpectralLineDatabase:
    cols = "freq", "A_ul", "E_low", "g_u"
    name_q_t = "Q_T"

    def __init__(self, fname):
        self.fname = fname
        self._data = {}

        conn = sqlite3.connect(fname)
        cursor = conn.cursor()
        query = "select * from partitionfunctions"
        cursor.execute(query)
        names = list(map(lambda x: x[0], cursor.description))
        self.x_T = np.array([float(name[3:].replace("_", ".")) for name in names[5:-6]])
        cursor.close()
        conn.close()

    def query(self, key, freq_list):
        prop_dict = self._load_data(key)
        data_ret = {key: [] for key in self.cols}
        data_ret["segment"] = []
        freqs = prop_dict["freq"]
        for i_segment, freq in enumerate(freq_list):
            freq_min = freq[0]
            idx_b = np.searchsorted(freqs, freq_min)
            freq_max = freq[-1]
            idx_e = np.searchsorted(freqs, freq_max)
            idx_e = min(idx_e, len(freqs) - 1)
            data_ret["segment"].append(np.full(idx_e - idx_b, i_segment))
            for col in self.cols:
                data_ret[col].append(prop_dict[col][idx_b:idx_e])
        data_ret["segment"] = np.concatenate(data_ret["segment"])
        for col in self.cols:
            data_ret[col] = np.concatenate(data_ret[col])
        data_ret[self.name_q_t] = self._data[key][self.name_q_t]
        data_ret["x_T"] = self.x_T
        return data_ret

    def query_specie_names(self, freq_list):
        """Find all entries in the given frequency ranges.

        Args:
            freq_list (list): A list of arrays to specify the frequencies to
                compute the spectral line model.

        Returns:
            list: A list of tuples (``specie_name``, ``transition_frequecy``).
        """
        conn = sqlite3.connect(self.fname)
        cursor = conn.cursor()

        query = f"select T_Name, T_Frequency from transitions where T_Frequency between ? and ?"
        data = []
        for freq in freq_list:
            cursor.execute(query, (freq[0], freq[-1]))
            data.extend(cursor.fetchall())
        return data

    def load_all_data(self):
        conn = sqlite3.connect(self.fname)
        cursor = conn.cursor()

        query = "select T_Name, T_Frequency, T_EinsteinA, T_EnergyLower, "\
            "T_UpperStateDegeneracy from transitions"
        data = defaultdict(lambda: {key: [] for key in self.cols})
        for line in cursor.execute(query):
            sub_dict = data[line[0]]
            sub_dict["freq"].append(line[1])
            sub_dict["A_ul"].append(line[2])
            sub_dict["E_low"].append(line[3])
            sub_dict["g_u"].append(line[4])

        for sub_dict in data.values():
            freq = np.asarray(sub_dict["freq"])
            inds = np.argsort(freq)
            sub_dict.update(
                freq=freq[inds],
                A_ul=np.asarray(sub_dict["A_ul"])[inds],
                E_low=np.asarray(sub_dict["E_low"])[inds]*1.438769, # Convert cm^-1 to K
                g_u=np.asarray(sub_dict["g_u"])[inds]
            )

        query = "select * from partitionfunctions"
        for line in cursor.execute(query):
            data[line[0]][self.name_q_t] = np.asarray(line[5:-6])

        cursor.close()
        conn.close()

        self._data = dict(data)

    def _load_data(self, key):
        if key in self._data:
            return self._data[key]

        conn = sqlite3.connect(self.fname)
        cursor = conn.cursor()

        query = "select T_Name, T_Frequency, T_EinsteinA, T_EnergyLower, "\
            "T_UpperStateDegeneracy from transitions where T_Name = ?"
        prop_dict = {"freq": [], "A_ul": [], "E_low": [], "g_u": []}
        for line in cursor.execute(query, (key,)):
            prop_dict["freq"].append(line[1])
            prop_dict["A_ul"].append(line[2])
            prop_dict["E_low"].append(line[3]*1.438769) # Convert cm^-1 to K
            prop_dict["g_u"].append(line[4])

        if len(prop_dict["freq"]) == 0:
            raise KeyError(f"Fail to find {key}.")

        query = "select * from partitionfunctions where PF_Name = ?"
        for line in cursor.execute(query, (key,)):
            tmp = [1. if val is None else val for val in line[5:-6]]
            prop_dict[self.name_q_t] = np.array(tmp)

        cursor.close()
        conn.close()

        self._data[key] = prop_dict
        return prop_dict
