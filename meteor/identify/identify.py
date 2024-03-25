import pickle
from copy import deepcopy

from ..algorithms import concat_identify_result, compute_T_single_data
from ..optimize import combine_mol_stores


def identify_file(idn, fname, config):
    data = pickle.load(open(fname, "rb"))
    res = idn.identify(
        data["mol_store"], config["sl_model"], data["params_best"],
    )
    return res


def identify_without_base(idn, dirname, config):
    res_list = []
    for fname in dirname.glob("*.pickle"):
        if is_exclusive(fname):
            continue
        data = pickle.load(open(fname, "rb"))
        res = idn.identify(
            data["mol_store"], config["sl_model"], data["params_best"],
        )
        if len(res.df_mol) == 0:
            continue
        res = res.extract_sub(data["mol_store"].mol_list[0]["id"])
        res_list.append(res)
    return concat_identify_result(res_list)


def identify_with_base(idn, dirname, fname_base, config):
    config_slm = config["sl_model"]

    data = pickle.load(open(fname_base, "rb"))
    mol_store_base = data["mol_store"]
    params_base = data["params_best"]
    T_single_dict_base = compute_T_single_data(
        mol_store_base, config_slm, params_base, data["freq"]
    )

    res_list = []
    for fname in dirname.glob("*.pickle"):
        if is_exclusive(fname):
            continue
        data = pickle.load(open(fname, "rb"))
        mol_store_combine, params_combine = combine_mol_stores(
            [mol_store_base, data["mol_store"]],
            [params_base, data["params_best"]],
            config["sl_model"]
        )
        T_single_dict = deepcopy(T_single_dict_base)
        T_single_dict.update(compute_T_single_data(
            data["mol_store"], config_slm, data["params_best"], data["freq"]
        ))
        res = idn.identify(
            mol_store_combine, config_slm, params_combine, T_single_dict
        )
        if len(res.df_mol) == 0:
            continue
        try:
            res = res.extract_sub(data["mol_store"].mol_list[0]["id"])
        except KeyError:
            res = None
        if res is not None:
            res_list.append(res)
    return concat_identify_result(res_list)


def is_exclusive(fname):
    name = str(fname.name)
    return name.startswith("identify") \
        or name.startswith("combine") \
        or name.startswith("tmp")