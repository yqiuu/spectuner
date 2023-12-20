import sys
import yaml
import pickle
from pathlib import Path
from multiprocessing import Pool

import numpy as np

from src.algorithms import select_molecules
from optimize_single import optimize


def main(config, config_trail):
    spec_obs = np.loadtxt(config["file_spec"])
    spec_obs = spec_obs[::-1] # Make freq ascending

    pool = Pool(config["opt_single"]["n_process"])
    FreqMin = spec_obs[0, 0]
    FreqMax = spec_obs[-1, 0]
    ElowMin = 0
    ElowMax = 2000.
    mol_dict = select_molecules(
        FreqMin, FreqMax, ElowMin, ElowMax,
        config["molecules"], config["elements"]
    )
    mol_name = config_trail["mol_name"]
    mol_dict = {mol_name: mol_dict[mol_name]}
    results = []
    for _ in range(config_trail["n_trail"]):
        results.append(optimize(spec_obs, mol_dict, config, pool))

    save_dir = Path(config["save_dir"])
    pickle.dump(results, open(save_dir/Path("{}.pickle".format(mol_name))))


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    config_trail = yaml.safe_load(open(sys.argv[2]))
    main(config, config_trail)