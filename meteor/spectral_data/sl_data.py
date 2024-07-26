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
        self.x_t = np.array([float(name[3:].replace("_", ".")) for name in names[5:-6]])
        cursor.close()
        conn.close()

    def query(self, key, freq_data):
        prop_dict = self._load_data(key)
        data_ret = {key: [] for key in self.cols}
        freqs = prop_dict["freq"]
        for freq in freq_data:
            freq_min = freq[0]
            idx_b = np.searchsorted(freqs, freq_min)
            freq_max = freq[-1]
            idx_e = np.searchsorted(freqs, freq_max)
            idx_e = min(idx_e, len(freqs) - 1)
            for col in self.cols:
                data_ret[col].append(prop_dict[col][idx_b:idx_e])
        for col in self.cols:
            data_ret[col] = np.concatenate(data_ret[col])
        data_ret[self.name_q_t] = self._data[key][self.name_q_t]
        return data_ret

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
            prop_dict["E_low"].append(line[3])
            prop_dict["g_u"].append(line[4])

        if len(prop_dict["freq"]) == 0:
            raise KeyError(f"Fail to find {key}.")

        query = "select * from partitionfunctions where PF_Name = ?"
        for line in cursor.execute(query, (key,)):
            prop_dict[self.name_q_t] = np.array(line[5:-6])

        cursor.close()
        conn.close()

        self._data[key] = prop_dict
        return prop_dict
