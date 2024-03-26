import pickle
from pathlib import Path
from multiprocessing import Pool

from .optimize import optimize
from ..preprocess import load_preprocess_select
from ..xclass_wrapper import MoleculeStore, Scaler
from ..identify.identify import identify
from ..fitting_model import create_fitting_model


__all__ = ["run_single"]


def run_single(config, parent_dir, need_identify=True):
    obs_data, mol_list, include_dict = load_preprocess_select(config)
    pool = Pool(config["opt_single"]["n_process"])

    id_offset = 0
    fname_base = config.get("fname_base", None)
    if fname_base is not None:
        data = pickle.load(open(fname_base, "rb"))
        base_data = data["T_pred"]
        for item in data["mol_store"].mol_list:
            id_offset = max(id_offset, item["id"])
        id_offset += 1
    else:
        base_data = None

    save_dir = Path(parent_dir)/"single"
    save_dir.mkdir(exist_ok=True)
    for item in mol_list:
        name = item["root"]
        item["id"] = item["id"] + id_offset
        mol_store = MoleculeStore([item], include_dict[name], Scaler())
        model = create_fitting_model(obs_data, mol_store, config, config["opt_single"], base_data)
        ret_dict = optimize(model, config["opt_single"], pool)
        pickle.dump(ret_dict, open(save_dir/Path("{}.pickle".format(name)), "wb"))

    if need_identify:
        identify(config, "single")