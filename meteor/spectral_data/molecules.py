import re
import warnings
from collections import defaultdict
from copy import deepcopy

try:
    from xclass import task_ListDatabase
except ImportError:
    warnings.warn("XCLASS is not installed.")

from .atoms import MolecularDecomposer


def select_molecules(freq_data, ElowMin, ElowMax, molecules,
                     elements=None, base_only=False,
                     iso_mode="combined", iso_order=1,
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

    normal_dict_list = []
    for freq in freq_data:
        normal_dict_list.append(group_by_normal_form(
            FreqMin=freq[0],
            FreqMax=freq[-1],
            ElowMin=ElowMin,
            ElowMax=ElowMax,
            moleclues=molecules,
            elements=elements,
            iso_mode=iso_mode,
            iso_order=iso_order,
            sort_mode=sort_mode,
            include_hyper=include_hyper,
            exclude_list=exclude_list_,
            rename_dict=rename_dict_
        ))

    # Merge all normal dict from different segment
    normal_dict_all = defaultdict(list)
    for normal_dict in normal_dict_list:
        for key, name_list in normal_dict.items():
            normal_dict_all[key].extend(name_list)
    # Remove duplicated moleclues in the normal dict
    for key in list(normal_dict_all.keys()):
        tmp = list(set(normal_dict_all[key]))
        tmp.sort()
        normal_dict_all[key] = tmp

    mol_list, master_name_dict \
        = replace_with_master_name(normal_dict_all, base_only)

    incldue_dict = defaultdict(lambda: [[] for _ in range(len(freq_data))])
    for i_segment, normal_dict in enumerate(normal_dict_list):
        for name, iso_list in normal_dict.items():
            master_name = master_name_dict[name]
            if master_name is not None:
                incldue_dict[master_name][i_segment]= deepcopy(iso_list)
    incldue_dict = dict(incldue_dict)

    return mol_list, incldue_dict


def group_by_normal_form(FreqMin, FreqMax, ElowMin, ElowMax,
                         moleclues, elements, iso_mode, iso_order,
                         sort_mode, include_hyper,
                         exclude_list, rename_dict):
    if exclude_list is None:
        exclude_list = []
    if rename_dict is None:
        rename_dict = {}

    contents = task_ListDatabase.ListDatabase(
        FreqMin, FreqMax, ElowMin, ElowMax,
        SelectMolecule=[], OutputDevice="quiet"
    )

    mol_names = choose_version(contents, exclude_list, sort_mode, include_hyper)
    mol_names = exclude_isotopes(mol_names, iso_order)
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


def choose_version(contents, exclude_list, sort_mode, include_hyper):
    counter = defaultdict(int)
    for item in contents:
        counter[item.split()[0]] += 1

    mol_dict = defaultdict(list)
    mol_dict_hyp = defaultdict(list)
    for name, num in counter.items():
        if name in exclude_list:
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