import pickle
from pathlib import Path
from multiprocessing import Pool

from .optimize import optimize
from ..preprocess import load_preprocess_select
from ..xclass_wrapper import MoleculeStore, Scaler
from ..identify import identify
from ..fitting_model import FittingModel


__all__ = ["run_single"]


def run_single(config, need_identify=True):
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

    for item in mol_list:
        name = item["root"]
        item["id"] = item["id"] + id_offset
        model = _create_model(name, obs_data, [item], include_dict, config, base_data)
        ret_dict = optimize(model, config["opt_single"], pool)
        save_dir = Path(config["save_dir"])/Path(config["opt_single"]["dirname"])
        pickle.dump(ret_dict, open(save_dir/Path("{}.pickle".format(name)), "wb"))

    if need_identify:
        identify(config, "single")


def _create_model(name, obs_data, mol_list_sub, include_dict, config, base_data):
    mol_store = MoleculeStore(mol_list_sub,  include_dict[name], Scaler())
    pm = mol_store.create_parameter_manager(config["sl_model"])
    # TODO: better way to create bounds?
    config_opt = config["opt_single"]
    bounds = pm.scaler.derive_bounds(
        pm,
        config_opt["bounds_mol"],
        config_opt["bounds_iso"],
        config_opt["bounds_misc"]
    )
    model = FittingModel(
        obs_data, mol_store, bounds, config["sl_model"],
        config_pm_loss=config.get("pm_loss", None),
        config_thr_loss=config.get("thr_loss", None),
        base_data=base_data
    )
    return model