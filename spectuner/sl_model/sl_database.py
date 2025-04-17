from __future__ import annotations
import sqlite3
import re
from collections import defaultdict
from typing import Optional
from abc import abstractmethod, ABC

import numpy as np
from astropy import constants, units

from .atoms import MolecularDecomposer
from ..peaks import create_spans, compute_shift


MHZ2KELIVN = ((constants.h/constants.k_B*units.MHz).to(units.Kelvin)).value


def query_species(sl_database, freq_list,
                  v_LSR=0., freqs_include=None, v_range=None,
                  species=None, elements=None,
                  collect_iso=True, iso_mode="combined", iso_order=1,
                  version_mode="default", include_hyper=False,
                  separate_all=False, exclude_list=None, rename_dict=None):
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
    entries = sl_database.query_transitions(freq_list)
    if freqs_include is not None:
        entries = check_freqs_include(
            entries, compute_shift(freqs_include, v_LSR), v_range
        )
    counter = defaultdict(int)
    for item in entries:
        counter[item[0]] += 1

    mol_tire = MolTrie()
    for entry in counter.keys():
        mol_tire.insert(MolRecord(entry, rename_dict))

    if species is None:
        records_include = mol_tire.search(())
    else:
        records_include = []
        for entry in species:
            record = MolRecord(entry, rename_dict)
            if collect_iso:
                prefix = record[:1]
            else:
                prefix = record
            records_include.extend(mol_tire.search(prefix))
    records_exclude = []
    for entry in exclude_list:
        records_exclude.extend(mol_tire.search(MolRecord(entry, rename_dict)))

    records = set(records_include) - set(records_exclude)
    records = sorted(records)
    names_include = [record.name for record in records]

    specie_names = choose_version(
        names_include, counter, version_mode, include_hyper
    )
    specie_names = exclude_isotopes(specie_names, iso_order)
    groups = derive_groups(
        specie_names, species, elements, iso_mode,
        exclude_list, rename_dict_, separate_all
    )
    return derive_specie_list(groups)


def prepare_specie_names(sl_database, freq_list,
                         v_LSR=0., freqs_include=None, v_range=None,
                         iso_order=1, version_mode="default",
                         include_hyper=False):
    entries = sl_database.query_transitions(freq_list)
    if freqs_include is not None:
        entries = check_freqs_include(
            entries, compute_shift(freqs_include, v_LSR), v_range
        )
    specie_names = choose_version(entries, version_mode, include_hyper)
    specie_names = exclude_isotopes(specie_names, iso_order)
    return specie_names


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


def derive_groups(specie_names, moleclues, elements, iso_mode,
                  exclude_list, rename_dict, separate_all):
    specie_data = derive_specie_data(specie_names, rename_dict)

    if moleclues is not None:
        fm_set, fm_root_set \
            = list(zip(*[derive_normal_formula(name, rename_dict)[:2] for name in moleclues]))
    exclude_list = decompose_exclude_list(exclude_list, rename_dict)

    # Filter elements and species
    if elements is not None:
        elements = set(elements)
        if "H" in elements:
            elements.add("D")
    groups = defaultdict(list)
    for fm_root, fm, mol_normal, atom_set, mol in specie_data:
        if check_exclude_list(mol_normal, exclude_list):
            continue

        if elements is not None and len(atom_set - elements) != 0:
            continue

        if iso_mode == "manual" and (moleclues is None or fm in fm_set):
            key = mol_normal if separate_all else fm
        elif iso_mode == "separate" and (moleclues is None or fm_root in fm_root_set):
            key = mol_normal if separate_all else fm
        elif iso_mode == "combined" and (moleclues is None or fm_root in fm_root_set):
            key = mol_normal if separate_all else fm_root
        else:
            continue
        groups[key].append(mol)

    return groups


def derive_specie_data(mol_list, rename_dict):
    specie_data = []
    for mol in mol_list:
        fm, fm_root, mol_normal = derive_normal_formula(mol, rename_dict)
        atom_set = derive_atom_set(fm_root)
        specie_data.append((fm_root, fm, mol_normal, atom_set, mol))
    return specie_data


def derive_normal_formula(mol_name, rename_dict):
    """Expand and rename an input formula.

    The function also checks isotopes in the formula.

    Args:
        mol_name (str): formula.
        rename_dict (dict): Rename dictionary.

    Returns:
        fm (str): Normal formula.
        fm_root (str): Normal formula with isotopes removed.
        mol_normal (str): Molecule name replaced with normal formula.
    """
    def expand(fm):
        """Expand formulae.
            HC3N > HCCCN
            CH3CN > CHHHCN
        """
        for pattern in re.findall("[A-Z][a-z]?\d", fm):
            fm = fm.replace(pattern, pattern[:-1]*int(pattern[-1]))
        return fm

    tmp = mol_name.split(";")
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
    tmp[0] = fm
    mol_normal = ";".join(tmp)
    return fm, fm_root, mol_normal


def derive_atom_set(fm):
    # Remove (Z-), E-, aGg-, AA-
    fm_ = fm
    for pattern in ["Z", "E", "GG", "GA", "AG", "AA", "G"]:
        fm_ = fm_.replace(pattern, "")

    atom_dict = MolecularDecomposer(fm_).ShatterFormula()
    atom_set = set(atom_dict.keys())
    return atom_set


def decompose_exclude_list(exclude_list, rename_dict):
    """Decompose an exclude_lsit, e.g.

    [CH3OH, HNCO;v=0, C2H3CN, H2C-13-CHCN;v=0;]
    -> [[CH3OH, C2H3CN], [HNCO;v=0], [H2C-13-CHCN;v=0;]]
    """
    exclude_list_new = [[], [], []]
    for name in exclude_list:
        mol_normal = derive_normal_formula(name, rename_dict)[2]
        num = len(mol_normal.split(";"))
        if num == 1:
            exclude_list_new[0].append(mol_normal)
        elif num == 2:
            exclude_list_new[1].append(mol_normal)
        else:
            exclude_list_new[2].append(mol_normal)
    return exclude_list_new


def check_exclude_list(name, exclude_list):
    tmp = name.split(";")
    if tmp[0] in exclude_list[0]:
        return True
    if ";".join(tmp[:2]) in exclude_list[1]:
        return True
    if name in exclude_list[2]:
        return True
    return False


def derive_specie_list(groups):
    idx = 0
    specie_list = []
    for species in groups.values():
        root_name = select_master_name(species)
        if root_name is None:
            continue
        specie_list.append({"id": idx, "root": root_name, "species": species})
        idx += 1
    return specie_list


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
    for pattern in re.findall("D\d", name):
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
            for pattern in re.findall("[A-Z][a-z]?\d", fm):
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
        node = self
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        records = []
        self._dfs(node, prefix, records)
        return records

    def _dfs(self, node, prefix, records):
        if node.record is not None:
            records.append(node.record)

        for char, child_node in node.children.items():
            self._dfs(child_node, (*prefix, char), records)


class SpectralLineDB(ABC):
    _cols = "freq", "A_ul", "E_low", "g_u"
    _q_min = 1e-10

    def __init__(self, freqs: list, names: list, cache: bool=False):
        inds = np.argsort(freqs)
        self._freqs = tuple(np.asarray(freqs)[inds])
        self._names = tuple(np.asarray(names)[inds])
        if cache:
            self._cache = {}
        else:
            self._cache = None

    @property
    def freqs(self) -> np.ndarray:
        """Frequencies of all entries."""
        return self._freqs

    @property
    def names(self) -> np.ndarray:
        """Names of all entries."""
        return self._names

    def query_transitions(self, freq_data: list) -> list:
        """Find all entries in the given frequency ranges.

        Args:
            freq_data (list): A list of arrays to specify the frequencies to
                compute the spectral line model. The code uses the ``min`` and
                ``max`` functions to find the minimum and maximum frequencies.

        Returns:
            list: A list of tuples (``specie_name``, ``transition_frequecy`` [MHz]).
        """
        data = []
        for freq in freq_data:
            # In some cases, freq is not ordered, so we need to use max and min
            freq_min = np.min(freq)
            idx_b = np.searchsorted(self.freqs, freq_min)
            freq_max = np.max(freq)
            idx_e = np.searchsorted(self.freqs, freq_max)
            data.extend(list(zip(self.names[idx_b:idx_e], self.freqs[idx_b:idx_e])))
        return data

    def query_sl_dict(self, key: str, freq_data: list, v_enlarge: float=0.) -> dict:
        # This function is almost the same as the one in sl_database.py
        sl_dict = self._load_sl_dict(key)
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
        if self._cache is not None and key in self._cache:
            return self._cache[key]

        sl_dict = self._load_sl_dict(key)
        self._cache[key] = sl_dict
        return sl_dict


class SQLSpectralLineDB(SpectralLineDB):
    def __init__(self, fname, cache=False):
        self._fname = fname

        conn = sqlite3.connect(fname)
        cursor = conn.cursor()

        query = """select T_Name, T_Frequency from transitions where """\
            """T_name not like '%RRL%'"""
        cursor.execute(query)
        data = cursor.fetchall()
        names, freqs = tuple(zip(*data))

        query = "select * from partitionfunctions"
        cursor.execute(query)
        cols = list(map(lambda x: x[0], cursor.description))
        self._x_T = np.array([float(col[3:].replace("_", ".")) for col in cols[5:-6]])

        cursor.close()
        conn.close()

        super().__init__(freqs, names, cache)

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
            tmp = [1. if val is None else val for val in line[5:-6]]
            sl_dict["Q_T"] = np.array(tmp)
        sl_dict["x_T"] = self._x_T

        cursor.close()
        conn.close()
        return sl_dict
