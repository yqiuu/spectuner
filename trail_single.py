import sys
import yaml
import pickle
from pathlib import Path
from multiprocessing import Pool

from src.preprocess import load_preprocess_select
from optimize_single import create_model, optimize


def main(config, config_trail):
    obs_data, mol_dict, mol_list, segment_dict = load_preprocess_select(config)
    pool = Pool(config["opt_single"]["n_process"])
    mol_name = config_trail["mol_name"]
    model = create_model(mol_name, obs_data, mol_dict, segment_dict, config)
    segments = segment_dict[mol_name]
    results = []
    for _ in range(config_trail["n_trail"]):
        results.append(optimize(model, mol_name, segments, config, pool))

    save_dir = Path(config_trail["save_dir"])
    pickle.dump(results, open(save_dir/Path("{}.pickle".format(mol_name)), "wb"))


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    config_trail = yaml.safe_load(open(sys.argv[2]))
    main(config, config_trail)