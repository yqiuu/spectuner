from __future__ import annotations
import sqlite3
import re
from typing import Optional, Literal
from abc import abstractmethod, ABC
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
from astropy import constants, units

from .atoms import MolecularDecomposer
from ..peaks import create_spans, compute_shift


MHZ2KELIVN = ((constants.h/constants.k_B*units.MHz).to(units.Kelvin)).value


def query_species(sl_db: SpectralLineDB,
                  freq_data: list,
                  v_LSR: float=0.,
                  freqs_include: Optional[tuple]=None,
                  v_range: Optional[tuple]=None,
                  species: Optional[tuple]=None,
                  elements: Optional[tuple]=None,
                  collect_iso: bool=True,
                  iso_order: int=2,
                  combine_iso: bool=False,
                  combine_state: bool=False,
                  version_mode: Literal["default", "most", "latest", "all"]="default",
                  include_hyper: bool=False,
                  exclude_list: Optional[tuple]=None,
                  rename_dict: Optional[dict]=None) -> list:
    """Query species from the database for fitting tasks.

    Args:
        sl_db: SpectralLineDB instance.
        freq_data: List of frequency ranges.
        v_LSR: LSR velocity [km/s].
        freqs_include: List of frequencies to include.
        v_range: Tolerance to check ``freqs_include``. Should be ``(min, max)``.
        species: List of species to include. If ``None``, inlcude all species.
        elements: List of elements to include. If ``None``, do not filter
            elements.
        collect_iso: If ``True``, collect isotopologues.
        iso_order: Number of isotopes to include (Do not use this).
        combine_iso: If ``True``, combine isotopologues and fitting them
            jointly.
        combine_state: If ``True``, combine states and fitting them jointly.
        include_hyper: If ``True``, include hyperfine states.
        exclude_list: List of species to exclude.
        rename_dict: A dict to rename species.

    Returns:
        List of species to fit.
    """
    if freqs_include is not None and len(freqs_include) == 0:
        return []

    rename_dict_ = {
        "NH2CN": "H2NCN",
        "H2CCHCN": "CH2CHCN",
        "CH2CHCN": "CH2CHCN",
        "C2H3CN": "CH2CHCN",
        "C2H5CN": "CH3CH2CN"
    }
    if rename_dict is not None:
        rename_dict_.update(**rename_dict)
    if exclude_list is None:
        exclude_list = []

    # Filter entries
    entries = sl_db.query_transitions(freq_data)
    if freqs_include is not None:
        entries = check_freqs_include(
            entries, compute_shift(freqs_include, v_LSR), v_range
        )
    counter = defaultdict(int)
    for item in entries:
        counter[item[0]] += 1
    counter = dict(counter)

    mol_tire = MolTrie()
    for entry in counter.keys():
        mol_tire.insert(MolRecord(entry, rename_dict_))

    if species is None:
        records_include = mol_tire.search(())
    else:
        records_include = []
        for entry in species:
            record = MolRecord(entry, rename_dict_)
            if collect_iso:
                prefix = list(record)
                prefix[1] = "?"
            else:
                prefix = record
            records_include.extend(mol_tire.search(prefix))
    records_exclude = []
    for entry in exclude_list:
        records_exclude.extend(mol_tire.search(MolRecord(entry, rename_dict_)))

    records = set(records_include) - set(records_exclude)

    # Filter elements
    if elements is not None:
        elements = set(elements)
        if "H" in elements:
            elements.add("D")
        records = filter(partial(check_elements, elements=elements), records)

    #
    record_dict = {record.name: record for record in records}
    specie_names = choose_version(
        record_dict.keys(), counter, version_mode, include_hyper
    )
    specie_names = exclude_isotopes(specie_names, iso_order)
    records = [record_dict[name] for name in specie_names]

    groups = derive_groups(records, combine_iso, combine_state)
    return groups, counter


def check_freqs_include(entries, freqs_include, v_range):
    """Include entries that are within the given frequecy ranges."""
    freqs = np.array([item[1] for item in entries])
    freqs = np.sort(freqs)
    spans = create_spans(freqs, *v_range)
    inds = np.searchsorted(freqs_include, spans[:, 0])
    cond = inds != len(freqs_include)
    inds[~cond] = len(freqs_include) - 1
    cond &= spans[:, 1] >= freqs_include[inds]
    return [item for idx, item in enumerate(entries) if cond[idx]]


def choose_version(names_incldue, counter, version_mode, include_hyper):
    mol_dict = defaultdict(list)
    mol_dict_hyp = defaultdict(list)
    for name in names_incldue:
        num = counter[name]
        tmp = name.split(";")
        # Ingore spin states or A/E states
        if tmp[-1].startswith("ortho") or tmp[-1].startswith("para") \
            or "A" in tmp[-1] or "E" in tmp[-1]:
            continue
        if tmp[2].startswith("hyp"):
            assert len(tmp) == 3, "Invalid entry: {}.".format(";".join(tmp))
            if "#" in tmp[-1]:
                tmp_a, tmp_b = tmp[-1].split("#")
                tmp[-1] = tmp_a
                tmp_b = f"#{tmp_b}"
            else:
                tmp_b = ""
            mol_dict_hyp[";".join(tmp)].append((tmp_b, num))
        else:
            mol_dict[";".join(tmp[:-1])].append((tmp[-1], num))

    if version_mode == "all":
        specie_names = []
        for prefix, post_list in mol_dict.items():
            for postfix in post_list:
                specie_names.append(";".join([prefix, postfix[0]]))
        #for prefix, post_list in mol_dict_hyp.items():
        #    for postfix in post_list:
        #        specie_names.append(";".join([prefix, postfix[0]]))
        return specie_names

    if version_mode == "default":
        sort_key = lambda item: item[0]
        is_reversed = False
    elif version_mode == "latest":
        sort_key = lambda item: item[0]
        is_reversed = True
    elif version_mode == "most":
        sort_key = lambda item: item[1] + int(item[0].split("#")[-1] if "#" in item[0] else 0)
        is_reversed = True
    else:
        raise ValueError("Unknown mode: {}.".format(version_mode))

    specie_names = []
    for prefix, post_list in mol_dict.items():
        post_list.sort(key=sort_key, reverse=is_reversed)
        specie_names.append(";".join([prefix, post_list[0][0]]))

    if not include_hyper:
        specie_names.sort()
        return specie_names

    for prefix, post_list in mol_dict_hyp.items():
        post_list.sort(key=sort_key, reverse=is_reversed)
        specie_names.append(prefix + post_list[0][0])
    specie_names.sort()
    return specie_names


def derive_groups(records, combine_iso, combine_state):
    groups = defaultdict(list)
    for record in records:
        # record (root, iso, state, version)
        if combine_iso and combine_state:
            key = record[0]
        elif combine_iso and not combine_state:
            key = (record[0], record[2])
        elif not combine_iso and combine_state:
            key = record[1]
        else:
            key = record
        groups[key].append(record.name)
    return list(groups.values())


def derive_atom_set(fm):
    # Remove (Z-), E-, aGg-, AA-
    fm_ = fm
    for pattern in ["Z", "E", "GG", "GA", "AG", "AA", "G"]:
        fm_ = fm_.replace(pattern, "")

    atom_dict = MolecularDecomposer(fm_).ShatterFormula()
    atom_set = set(atom_dict.keys())
    return atom_set


def check_elements(record, elements):
    atom_set = derive_atom_set(record[0])
    return len(atom_set - elements) == 0


def select_master_name(name_list):
    def sort_key(name):
        s = ""
        if is_ground_state(name):
            s += "0"
        else:
            s += "1"
        if has_isotope(name):
            s += "1"
        else:
            s += "0"
        if is_hyper_state(name):
            s += "1"
        else:
            s += "0"
        return f"{s}{name}"

    name_list_ = name_list.copy()
    name_list_.sort(key=sort_key)
    return name_list_[0]


def exclude_isotopes(specie_names, iso_order):
    return [name for name in specie_names if count_iso_atoms(name) <= iso_order]


def count_iso_atoms(name):
    # TODO: This function cannot handle complex formulae, e.g. (C-13-HOH)2
    name = name.split(";")[0]
    for pattern in re.findall(r"D\d", name):
        name = name.replace(pattern, pattern[:-1]*int(pattern[-1]))
    return name.count("D") + len(re.findall('-([0-9])([0-9])[-]?', name))


def is_ground_state(name):
    return name.split(";")[1] == "v=0"


def is_hyper_state(name):
    return name.split(";")[2].startswith("hyp")


def has_isotope(name):
    pattern = r"-([0-9])([0-9])[-]?"
    return re.search(pattern, name) is not None or "D" in name


def latex_mol_formula(name):
    """Convert the input name into latex format."""
    def replace_isotope(match):
        element = match.group(1)
        num = match.group(2)
        return f"$^{{{num}}}${element}"

    def replace_number(match):
        element = match.group(1)
        num = match.group(2)
        return f"{element}$_{{{num}}}$"

    name_ret = re.sub(r'([A-Z][a-z]?)-([0-9]+)-?', replace_isotope, name)
    name_ret = re.sub(r'([A-Z][a-z]?)([0-9]+)', replace_number, name_ret)
    name_ret = re.sub(r'(\([^)]+\))([0-9]+)', replace_number, name_ret)
    return name_ret


def create_spectral_line_db(fname: str, cache: bool=False):
    return SQLSpectralLineDB(fname, cache=cache)


class MolRecord(tuple):
    """
    Args:
        entry: Entry name.
        rename_dict: Rename dictionary.
    """
    def __new__(cls, entry: str, rename_dict: Optional[dict]=None) -> None:
        def expand(fm):
            """Expand formulae.
                HC3N > HCCCN
                CH3CN > CHHHCN
            """
            for pattern in re.findall(r"[A-Z][a-z]?\d", fm):
                fm = fm.replace(pattern, pattern[:-1]*int(pattern[-1]))
            return fm

        if rename_dict is None:
            rename_dict = {}

        tmp = entry.split(";")
        fm = tmp[0]
        if fm in rename_dict:
            fm = rename_dict[fm]

        # Remove iso
        pattern = r"-([0-9])([0-9])[-]?"
        fm_root = re.sub(pattern, "", fm)
        fm_root = fm_root.replace("D", "H")

        #
        fm = expand(fm)
        fm_root = expand(fm_root)

        instance = super().__new__(cls, (fm_root, fm, *tmp[1:]))
        instance.name = entry
        return instance

    def __repr__(self) -> str:
        return "{} -> {}".format(self.name, super().__repr__())


class MolTrie:
    def __init__(self):
        self.children = {}
        self.record = None

    def insert(self, record):
        node = self
        for char in record:
            if char not in node.children:
                node.children[char] = MolTrie()
            node = node.children[char]
        node.record = record

    def search(self, prefix):
        records = []
        self._dfs(self, prefix, (), records)
        return records

    def _dfs(self, node, prefix, current_path, records):
        if not prefix:
            if node.record is not None:
                records.append(node.record)
            for child_char, child_node in node.children.items():
                self._dfs(child_node, prefix, (*current_path, child_char), records)
            return

        char = prefix[0]

        if char == "?":
            for child_char, child_node in node.children.items():
                self._dfs(child_node, prefix[1:], (*current_path, child_char), records)
        else:
            if char in node.children:
                self._dfs(node.children[char], prefix[1:], (*current_path, char), records)


class SpectralLineDB(ABC):
    _cols = "freq", "A_ul", "E_low", "g_u"
    _q_min = 1e-10

    def __init__(self, cache: bool=False):
        self._cache = cache
        self._data = {}

    def query_transitions(self, freq_data: list) -> list:
        """Find all entries in the given frequency ranges.

        Args:
            freq_data (list): A list of arrays to specify the frequencies to
                compute the spectral line model. The code uses the ``min`` and
                ``max`` functions to find the minimum and maximum frequencies.

        Returns:
            list: A list of tuples (``specie_name``, ``transition_frequecy`` [MHz]).
        """
        names, freqs = self.load_all_transitions()
        inds = np.argsort(freqs)
        names = names[inds]
        freqs = freqs[inds]

        data = []
        for freq in freq_data:
            # In some cases, freq is not ordered, so we need to use max and min
            freq_min = np.min(freq)
            idx_b = np.searchsorted(freqs, freq_min)
            freq_max = np.max(freq)
            idx_e = np.searchsorted(freqs, freq_max)
            data.extend(list(zip(names[idx_b:idx_e], freqs[idx_b:idx_e])))
        return data

    @abstractmethod
    def load_all_transitions(self):
        """Load all transitions.

        Returns:
            tuple:  (names, freqs).
        """

    def query_sl_dict(self, key: str, freq_data: list, v_enlarge: float=0.) -> dict:
        # This function is almost the same as the one in sl_database.py
        sl_dict = self._load_sl_dict_with_cache(key)
        data_ret = {key: [] for key in self._cols}
        data_ret["segment"] = []
        freqs = sl_dict["freq"]
        for i_segment, freq in enumerate(freq_data):
            # In some cases, freq is not ordered, so we need to use max and min
            freq_min = compute_shift(np.min(freq), -v_enlarge)
            idx_b = np.searchsorted(freqs, freq_min)
            freq_max = compute_shift(np.max(freq), v_enlarge)
            idx_e = np.searchsorted(freqs, freq_max)
            data_ret["segment"].append(np.full(idx_e - idx_b, i_segment))
            for col in self._cols:
                data_ret[col].append(sl_dict[col][idx_b:idx_e])
        data_ret["segment"] = np.concatenate(data_ret["segment"])
        for col in self._cols:
            data_ret[col] = np.concatenate(data_ret[col])
        data_ret["E_up"] = data_ret["freq"]*MHZ2KELIVN + data_ret["E_low"]
        data_ret["Q_T"] = np.clip(sl_dict["Q_T"], self._q_min, None)
        data_ret["x_T"] = sl_dict["x_T"]
        return data_ret

    @abstractmethod
    def _load_sl_dict(self, key: str) -> dict:
        """Load the spectral line data of the input key."""

    def _load_sl_dict_with_cache(self, key: str) -> dict:
        if key in self._data:
            return self._data[key]

        sl_dict = self._load_sl_dict(key)
        if self._cache:
            self._data[key] = sl_dict
        return sl_dict


class SQLSpectralLineDB(SpectralLineDB):
    def __init__(self, fname, cache=False):
        super().__init__(cache)
        self._fname = fname
        # load x_T
        conn = sqlite3.connect(self._fname)
        cursor = conn.cursor()
        query = "select * from partitionfunctions"
        cursor.execute(query)
        cols = list(map(lambda x: x[0], cursor.description))
        idx_b = cols.index("PF_1_072")
        idx_e = cols.index("PF_1000_000") + 1
        self._slice_pf = slice(idx_b, idx_e)
        self._x_T = np.array([float(col[3:].replace("_", "."))
                              for col in cols[self._slice_pf]])
        cursor.close()
        conn.close()

    def load_all_transitions(self):
        try:
            return self._transitions
        except AttributeError:
            pass

        conn = sqlite3.connect(self._fname)
        query = """select T_Name, T_Frequency from transitions where """\
            """T_name not like '%RRL%'"""
        df = pd.read_sql_query(query, conn)
        names = df['T_Name'].to_numpy()
        freqs = df['T_Frequency'].to_numpy()
        conn.close()

        if self._cache:
            self._transitions = names, freqs

        return names, freqs

    def _load_sl_dict(self, key: str) -> dict:
        conn = sqlite3.connect(self._fname)
        cursor = conn.cursor()

        query = "select T_Name, T_Frequency, T_EinsteinA, T_EnergyLower, "\
            "T_UpperStateDegeneracy from transitions where T_Name = ?"
        sl_dict = {"freq": [], "A_ul": [], "E_low": [], "g_u": []}
        for line in cursor.execute(query, (key,)):
            sl_dict["freq"].append(line[1])
            sl_dict["A_ul"].append(line[2])
            sl_dict["E_low"].append(line[3]*1.438769) # Convert cm^-1 to K
            sl_dict["g_u"].append(line[4])

        if len(sl_dict["freq"]) == 0:
            raise KeyError(f"Fail to find {key}.")

        query = "select * from partitionfunctions where PF_Name = ?"
        for line in cursor.execute(query, (key,)):
            tmp = [1. if val is None else val for val in line[self._slice_pf]]
            sl_dict["Q_T"] = np.array(tmp)
        sl_dict["x_T"] = self._x_T
        cursor.close()
        conn.close()
        return sl_dict
