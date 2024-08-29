import re
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
try:
    from xclass import task_ListDatabase
except ImportError:
    warnings.warn("XCLASS is not installed.")

from .atoms import MolecularDecomposer
from ..peaks import create_spans, compute_shift


def query_molecules(freq_data, ElowMin, ElowMax,
                    v_LSR=0., freqs_include=None, v_range=None,
                    molecules=None, elements=None,
                    base_only=False, iso_mode="combined", iso_order=1,
                    sort_mode="default", include_hyper=False,
                    separate_all=False, exclude_list=None, rename_dict=None):
    """Select possible molecules in the given frequency range.

    Args:
        iso_mode (str): The way to deal with isotoplogues and states.
            - 'combine': Collect all possible isotoplogues and fit them jointly.
            - 'separate': Collect all possible isotoplogues and fit them
            separately.
            - 'manual': Only collect isotoplogues given by ``molecules``.

    """
    rename_dict_ = {
        "NH2CN": "H2NCN",
        "H2CCHCN-15": "CH2CHCN-15",
        "C2H3CN": "CH2CHCN",
        "C2H5CN": "CH3CH2CN"
    }
    if rename_dict is not None:
        rename_dict_.update(**rename_dict)
    exclude_list_ = [
        "HNC3;v=0;", # Duplicated
        "H2C-13-CHCN;v=0;", # Duplicated
        "H2CC-13-HCN;v=0;", # Duplicated
        "H2CCHC-13-N;v=0;", # Duplicated
    ]
    if exclude_list is not None:
        exclude_list_.extend(exclude_list)

    if not separate_all:
        mol_names_tot = prepare_mol_names(
            freq_data, ElowMin, ElowMax, v_LSR, freqs_include, v_range,
            iso_order, sort_mode, include_hyper
        )
        normal_dict = group_by_normal_form(
            mol_names_tot, molecules, elements, iso_mode, exclude_list_, rename_dict_
        )
        mol_list, master_name_dict = replace_with_master_name(normal_dict, base_only)

        # prepare include_dict
        normal_dict_list = []
        for freqs in freq_data:
            mol_names = prepare_mol_names(
                [freqs], ElowMin, ElowMax, v_LSR, freqs_include, v_range,
                iso_order, sort_mode, include_hyper, include_all=True
            )
            normal_dict_list.append(group_by_normal_form(
                mol_names, molecules, elements, iso_mode, exclude_list_, rename_dict_
            ))
        incldue_dict = defaultdict(lambda: [[] for _ in range(len(freq_data))])
        for i_segment, normal_dict in enumerate(normal_dict_list):
            for name, iso_list in normal_dict.items():
                if name not in master_name_dict:
                    continue
                iso_list_new = []
                for iso in iso_list:
                    if iso in mol_names_tot:
                        iso_list_new.append(iso)
                master_name = master_name_dict[name]
                if master_name is not None:
                    incldue_dict[master_name][i_segment]= iso_list_new
        incldue_dict = dict(incldue_dict)

        return mol_list, incldue_dict

    mol_names = prepare_mol_names(
        freq_data, ElowMin, ElowMax, v_LSR, freqs_include, v_range,
        iso_order, sort_mode, include_hyper
    )
    mol_names = filter_mol_names(mol_names, molecules, iso_mode, exclude_list_, rename_dict_)
    mol_list = []
    for idx, name in enumerate(mol_names):
        mol_list.append({"id": idx, "root": name, "molecules": [name]})

    # prepare include_dict
    mol_names_list = []
    for freqs in freq_data:
        mol_names = prepare_mol_names(
            [freqs], ElowMin, ElowMax, v_LSR, freqs_include, v_range,
            iso_order, sort_mode, include_hyper
        )
        mol_names = filter_mol_names(mol_names, molecules, iso_mode, exclude_list_, rename_dict_)
        mol_names_list.append(mol_names)
    incldue_dict = defaultdict(lambda: [[] for _ in range(len(freq_data))])
    for i_segment, mol_names in enumerate(mol_names_list):
        for name in mol_names:
            incldue_dict[name][i_segment].append(name)
    incldue_dict = dict(incldue_dict)

    return mol_list, incldue_dict


def group_by_normal_form(mol_names, moleclues, elements, iso_mode,
                         exclude_list, rename_dict):
    mol_data = derive_mol_data(mol_names, rename_dict)

    if moleclues is not None:
        fm_set, fm_root_set \
            = list(zip(*[derive_normal_formula(name, rename_dict)[:2] for name in moleclues]))
    exclude_list = decompose_exclude_list(exclude_list, rename_dict)

    # Filter elements and molecules
    if elements is not None:
        elements = set(elements)
        if "H" in elements:
            elements.add("D")
    normal_dict = defaultdict(list)
    for fm_root, fm, mol_normal, atom_set, mol in mol_data:
        if check_exclude_list(mol_normal, exclude_list):
            continue

        if elements is not None and len(atom_set - elements) != 0:
            continue

        if iso_mode == "manual" and (moleclues is None or fm in fm_set):
            normal_dict[fm].append(mol)
            continue

        if iso_mode == "separate" and (moleclues is None or fm_root in fm_root_set):
            normal_dict[fm].append(mol)
            continue

        if iso_mode == "combined" and (moleclues is None or fm_root in fm_root_set):
            normal_dict[fm_root].append(mol)
            continue

    return normal_dict


def prepare_mol_names(freq_data, E_low_min, E_low_max,
                      v_LSR=0., freqs_include=None, v_range=None,
                      iso_order=1, sort_mode="default",
                      include_hyper=False, include_all=False):
    contents = []
    for freqs in freq_data:
        freq_min = compute_shift(freqs[0], v_LSR)
        freq_max = compute_shift(freqs[-1], v_LSR)
        contents.extend(task_ListDatabase.ListDatabase(
            freq_min, freq_max, E_low_min, E_low_max,
            SelectMolecule=[], OutputDevice="quiet"
        ))
    if freqs_include is not None:
        contents = check_spans(
            contents, compute_shift(freqs_include, v_LSR), v_range
        )

    mol_names = choose_version(contents, sort_mode, include_hyper, include_all)
    mol_names = exclude_isotopes(mol_names, iso_order)
    return mol_names


def filter_mol_names(mol_names, moleclues, iso_mode, exclude_list, rename_dict):
    if moleclues is not None:
        fm_set, fm_root_set \
            = list(zip(*[derive_normal_formula(name, rename_dict)[:2] for name in moleclues]))
    mol_names_ret = []
    for name in mol_names:
        fm, fm_root, mol_normal = derive_normal_formula(name, rename_dict)
        if check_exclude_list(mol_normal, exclude_list):
            continue

        if iso_mode == "manual" and (moleclues is None or fm in fm_set):
            mol_names_ret.append(name)
            continue

        if iso_mode in ["combined", "separate"] and (moleclues is None or fm_root in fm_root_set):
            mol_names_ret.append(name)
            continue
    return mol_names_ret


def check_spans(contents, freqs, v_range):
    freqs_mol = np.array([float(line.split()[3]) for line in contents])
    freqs_mol = np.sort(freqs_mol)
    spans = create_spans(freqs_mol, *v_range)
    inds = np.searchsorted(freqs, spans[:, 0])
    cond = inds != len(freqs)
    inds[~cond] = len(freqs) - 1
    cond &= spans[:, 1] >= freqs[inds]
    return [item for idx, item in enumerate(contents) if cond[idx]]


def choose_version(contents, sort_mode, include_hyper, include_all):
    counter = defaultdict(int)
    for item in contents:
        counter[item.split()[0]] += 1

    mol_dict = defaultdict(list)
    mol_dict_hyp = defaultdict(list)
    for name, num in counter.items():
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

    if include_all:
        mol_names = []
        for prefix, post_list in mol_dict.items():
            for postfix in post_list:
                mol_names.append(";".join([prefix, postfix[0]]))
        for prefix, post_list in mol_dict_hyp.items():
            for postfix in post_list:
                mol_names.append(";".join([prefix, postfix[0]]))
        return mol_names

    if sort_mode == "default":
        sort_key = lambda item: item[0]
        is_reversed = False
    elif sort_mode == "latest":
        sort_key = lambda item: item[0]
        is_reversed = True
    elif sort_mode == "most":
        sort_key = lambda item: item[1] + int(item[0].split("#")[-1] if "#" in item[0] else 0)
        is_reversed = True
    else:
        raise ValueError("Unknown mode: {}.".format(sort_mode))

    mol_names = []
    for prefix, post_list in mol_dict.items():
        post_list.sort(key=sort_key, reverse=is_reversed)
        mol_names.append(";".join([prefix, post_list[0][0]]))

    if not include_hyper:
        mol_names.sort()
        return mol_names

    for prefix, post_list in mol_dict_hyp.items():
        post_list.sort(key=sort_key, reverse=is_reversed)
        mol_names.append(prefix + post_list[0][0])
    mol_names.sort()
    return mol_names


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


def exclude_isotopes(mol_names, iso_order):
    return [name for name in mol_names if count_iso_atoms(name) <= iso_order]


def derive_mol_data(mol_list, rename_dict):
    mol_data = []
    for mol in mol_list:
        fm, fm_root, mol_normal = derive_normal_formula(mol, rename_dict)
        atom_set = derive_atom_set(fm_root)
        mol_data.append((fm_root, fm, mol_normal, atom_set, mol))
    return mol_data


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


def derive_normal_mol_name(mol_name, fm_normal):
    tmp = mol_name.split(";")
    tmp[0] = fm_normal
    return ";".join(tmp)


def replace_with_master_name(normal_dict, base_only):
    mol_dict = defaultdict(list)
    master_name_dict = {}
    for normal_name, name_list in normal_dict.items():
        name_list.sort()
        master_name = select_master_name(name_list)
        master_name_dict[normal_name] = master_name
        if master_name is None:
            continue
        for name in name_list:
            if base_only and is_ground_state(name):
                continue
            if name == master_name:
                mol_dict[master_name]
            else:
                mol_dict[master_name].append(name)
    mol_list = []
    for idx, (name, mols) in enumerate(mol_dict.items()):
        item = {"id": idx, "root": name}
        mols_new = [name]
        mols_new.extend(mols)
        item["molecules"] = mols_new
        mol_list.append(item)
    return mol_list, master_name_dict


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


def count_iso_atoms(name):
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