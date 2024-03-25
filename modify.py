import sys
import yaml
import pickle
from pathlib import Path

from src.optimize import combine_mol_stores


def modify(config, config_modify):
    config_slm = config["sl_model"]
    exclude_id_list = config_modify["exclude_id_list"]
    exclude_name_set = set(config_modify["exclude_name_list"])

    save_dir = Path(config["save_dir"])/Path(config["opt_combine"]["dirname"])
    save_name = save_dir/Path("combine_final.pickle")
    data_combine = pickle.load(open(save_dir/Path("combine.pickle"), "rb"))
    freq_data = data_combine["freq"]
    mol_store = data_combine["mol_store"]

    names_in = []
    for item in mol_store.mol_list:
        mols = []
        if item["id"] not in exclude_id_list:
            mols.extend(item["molecules"])
        mols = [name for name in mols if name not in exclude_name_set]
        names_in.extend(mols)

    mol_store_new, params_new = mol_store.select_subset_with_params(
        names_in, data_combine["params_best"], config_slm
    )

    include_id_list = config_modify["include_id_list"]
    if len(include_id_list) != 0:
        mol_store_list = [mol_store_new]
        params_list = [params_new]
        data_list = load_data_list(save_dir, include_id_list)
        for data in data_list:
            mol_store = data["mol_store"]
            params = data["params_best"]
            names_in = set(mol_store.mol_list[0]["molecules"]) - exclude_name_set
            mol_store, params \
                = mol_store.select_subset_with_params(names_in, params, config_slm)
            mol_store_list.append(mol_store)
            params_list.append(params)
        mol_store_new, params_new \
            = combine_mol_stores(mol_store_list, params_list, config_slm)

    T_pred_data = mol_store_new.compute_T_pred_data(params_new, freq_data, config_slm)
    save_dict = {
        "mol_store": mol_store_new,
        "freq": freq_data,
        "T_pred": T_pred_data,
        "params_best": params_new
    }
    pickle.dump(save_dict, open(save_name, "wb"))


def load_data_list(target_dir, include_id_list):
    data_list = []
    for fname in target_dir.glob("*.pickle"):
        key = str(fname.name).split("_")[0]
        try:
            key = int(key)
        except:
            continue
        if key in include_id_list:
            data_list.append(pickle.load(open(fname, "rb")))
    return data_list


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    config_modify = yaml.safe_load(open(sys.argv[2]))
    modify(config, config_modify)