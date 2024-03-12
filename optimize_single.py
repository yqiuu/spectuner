import sys
import yaml
import pickle
from pathlib import Path
from multiprocessing import Pool

from src.preprocess import load_preprocess_select
from src.xclass_wrapper import MoleculeStore, Scaler
from src.fitting_model import FittingModel
from src.optimize import optimize


def main(config):
    obs_data, mol_list, include_dict = load_preprocess_select(config)
    pool = Pool(config["opt_single"]["n_process"])
    for item in mol_list:
        name = item["root"]
        model = create_model(name, obs_data, [item], include_dict, config)
        ret_dict = optimize(model, config["opt_single"], pool)
        save_dir = Path(config["save_dir"])
        pickle.dump(ret_dict, open(save_dir/Path("{}.pickle".format(name)), "wb"))


def create_model(name, obs_data, mol_list_sub, include_dict, config):
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
    fname_base = config.get("fname_base", None)
    if fname_base is not None:
        base_data = pickle.load(open(fname_base, "rb"))["T_pred"]
    else:
        base_data = None
    model = FittingModel(
        obs_data, mol_store, bounds, config["sl_model"],
        config_pm_loss=config_opt.get("pm_loss", None),
        config_thr_loss=config_opt.get("thr_loss", None),
        base_data=base_data
    )
    return model


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    main(config)