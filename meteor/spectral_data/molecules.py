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
from ..identify import create_spans, compute_shift


def query_molecules(freq_data, ElowMin, ElowMax,
                    v_LSR=0., freqs_include=None, v_range=None,
                    molecules=None, elements=None,
                    base_only=False, iso_mode="combined", iso_order=1,
                    sort_mode="largest", include_hyper=False,
                    exclude_list=None, rename_dict=None):
    rename_dict_ = {
        "NH2CN": "H2NCN",
        "H2CCHCN-15": "CH2CHCN-15",
        "C2H3CN": "CH2CHCN",
        "C-13-H3C-13-H2C-13-N": "C2H5CN",
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

    mol_names = prepare_mol_names(
        freq_data, ElowMin, ElowMax, v_LSR, freqs_include, v_range,
        iso_order, sort_mode, include_hyper, exclude_list_
    )
    normal_dict = group_by_normal_form(mol_names, molecules, elements, iso_mode, rename_dict_)
    mol_list, master_name_dict = replace_with_master_name(normal_dict, base_only)

    # prepare include_dict
    normal_dict_list = []
    for freqs in freq_data:
        mol_names = prepare_mol_names(
            [freqs], ElowMin, ElowMax, v_LSR, freqs_include, v_range,
            iso_order, sort_mode, include_hyper, exclude_list_
        )
        normal_dict_list.append(
            group_by_normal_form(mol_names, molecules, elements, iso_mode, rename_dict_)
        )
    incldue_dict = defaultdict(lambda: [[] for _ in range(len(freq_data))])
    for i_segment, normal_dict in enumerate(normal_dict_list):
        for name, iso_list in normal_dict.items():
            if name not in master_name_dict:
                continue
            master_name = master_name_dict[name]
            if master_name is not None:
                incldue_dict[master_name][i_segment]= deepcopy(iso_list)
    incldue_dict = dict(incldue_dict)

    return mol_list, incldue_dict


def group_by_normal_form(mol_names, moleclues, elements, iso_mode, rename_dict):
    mol_data = derive_mol_data(mol_names, rename_dict)

    if moleclues is not None:
        fm_root_set, fm_set \
            = list(zip(*[derive_normal_form(name, rename_dict)[:2] for name in moleclues]))

    # Filter elements and molecules
    if elements is not None:
        elements = set(elements)
        if "H" in elements:
            elements.add("D")
    normal_dict = defaultdict(list)
    for fm_root, fm, atom_set, mol in mol_data:
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
                      iso_order=1, sort_mode="largest",
                      include_hyper=False, exclude_list=None):
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

    mol_names = choose_version(contents, exclude_list, sort_mode, include_hyper)
    mol_names = exclude_isotopes(mol_names, iso_order)
    return mol_names


def check_spans(contents, freqs, v_range):
    freqs_mol = np.array([float(line.split()[3]) for line in contents])
    freqs_mol = np.sort(freqs_mol)
    spans = create_spans(freqs_mol, *v_range)
    inds = np.searchsorted(freqs, spans[:, 0])
    cond = inds != len(freqs)
    inds[~cond] = len(freqs) - 1
    cond &= spans[:, 1] >= freqs[inds]
    return [item for idx, item in enumerate(contents) if cond[idx]]


def choose_version(contents, exclude_list, sort_mode, include_hyper):
    counter = defaultdict(int)
    for item in contents:
        counter[item.split()[0]] += 1

    exclude_list = decompose_exclude_list(exclude_list)
    mol_dict = defaultdict(list)
    mol_dict_hyp = defaultdict(list)
    for name, num in counter.items():
        if check_exclude_list(name, exclude_list):
            continue
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

    if sort_mode == "default":
        sort_key = lambda item: item[0]
        is_reversed = False
    elif sort_mode == "latest":
        sort_key = lambda item: item[0]
        is_reversed = True
    elif sort_mode == "largest":
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


def decompose_exclude_list(exclude_list):
    """Decompose an exclude_lsit, e.g.

    [CH3OH, HNCO;v=0, C2H3CN, H2C-13-CHCN;v=0;]
    -> [[CH3OH, C2H3CN], [HNCO;v=0], [H2C-13-CHCN;v=0;]]
    """
    exclude_list_new = [[], [], []]
    for name in exclude_list:
        num = len(name.split(";"))
        if num == 1:
            exclude_list_new[0].append(name)
        elif num == 2:
            exclude_list_new[1].append(name)
        else:
            exclude_list_new[2].append(name)
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
        name = mol.split(";")[0] # HCCCN;v=0; > HCCCN
        fm_root, fm, atom_set = derive_normal_form(name, rename_dict)
        mol_data.append((fm_root, fm, atom_set, mol))
    return mol_data


def derive_normal_form(mol_name, rename_dict):
    def expand(fm):
        """Expand formulae.
            HC3N > HCCCN
            CH3CN > CHHHCN
        """
        for pattern in re.findall("[A-Z][a-z]?\d", fm):
            fm = fm.replace(pattern, pattern[:-1]*int(pattern[-1]))
        return fm

    fm, *_ = mol_name.split(";")
    if fm in rename_dict:
        fm = rename_dict[fm]

    # Remove iso
    pattern = r"-([0-9])([0-9])[-]?"
    fm_root = re.sub(pattern, "", fm)
    fm_root = fm_root.replace("D", "H")

    #
    fm = expand(fm)
    fm_root = expand(fm_root)

    atom_dict = MolecularDecomposer(fm_root).ShatterFormula()
    atom_set = set(atom_dict.keys())

    return fm_root, fm, atom_set


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
    name_list_1 = tuple(filter(is_ground_state, name_list))
    if len(name_list_1) == 1:
        return name_list_1[0]
    if len(name_list_1) > 1:
        name_list_2 = tuple(filter(lambda name: not has_isotope(name), name_list_1))
        if len(name_list_2) == 0:
            return name_list_1[0]
        if len(name_list_2) == 1:
            return name_list_2[0]
        if len(name_list_2) > 1:
            name_list_3 = tuple(filter(lambda name: not is_hyper_state(name), name_list_2))
            if len(name_list_3) == 0:
                return name_list_2[0]
            if len(name_list_3) == 1:
                return name_list_3[0]
            raise ValueError("Multiple master name", name_list_2)


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