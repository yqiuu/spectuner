import sys
import yaml
import pickle
from pathlib import Path
from multiprocessing import Pool

from src.preprocess import load_preprocess_select
from src.fitting_model import create_fitting_model
from src.optimize import optimize


def main(config):
    obs_data, mol_list, segment_dict, include_dict = load_preprocess_select(config)
    pool = Pool(config["opt_single"]["n_process"])
    print(mol_list)
    for item in mol_list:
        name = item["root"]
        model = create_model(name, obs_data, [item], segment_dict, include_dict, config)
        segments = segment_dict[name]
        ret_dict = optimize(model, name, segments, config["opt_single"], pool)
        save_dir = Path(config["save_dir"])
        pickle.dump(ret_dict, open(save_dir/Path("{}.pickle".format(name)), "wb"))


def create_model(name, obs_data, mol_list_sub, segment_dict, include_dict, config):
    obs_data_sub = []
    for idx in segment_dict[name]:
        obs_data_sub.append(obs_data[idx])
    include_list = include_dict[name]
    fname_base = config.get("fname_base", None)
    if fname_base is not None:
        base_data = pickle.load(open(fname_base, "rb"))["T_pred"]
    else:
        base_data = None
    model = create_fitting_model(
        obs_data_sub, mol_list_sub, include_list,
        config["xclass"], config["opt_single"],
        base_data=base_data
    )
    return model


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    main(config)