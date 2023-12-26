import glob
import numpy as np

from .algorithms import select_molecules, select_molecules_multi


def load_spectra(config):
    file_spec = config["file_spec"]
    ElowMin = config["ElowMin"]
    ElowMax = config["ElowMax"]
    temp_back = config["xclass"].get("tBack", 0.)

    if isinstance(file_spec, str) and file_spec.startswith("glob:"):
        file_spec = glob.glob(file_spec.replace("glob:", ""))

    if isinstance(file_spec, str):
        obs_data = np.loadtxt(file_spec)
        obs_data = preprocess_spectrum(obs_data, temp_back)
        FreqMin = obs_data[0, 0]
        FreqMax = obs_data[-1, 0]
        mol_dict = select_molecules(
            FreqMin, FreqMax, ElowMin, ElowMax,
            config["molecules"], config["elements"], config["base_only"]
        )
        segment_dict = None
        mol_list = list(mol_dict.keys())
    else:
        obs_data = [np.loadtxt(fname) for fname in file_spec]
        obs_data = [preprocess_spectrum(spec, temp_back) for spec in obs_data]
        mol_dict, segment_dict = select_molecules_multi(
            obs_data, ElowMin, ElowMax,
            config["molecules"], config["elements"], config["base_only"]
        )
        mol_list = list(segment_dict.keys())
    return obs_data, mol_dict, mol_list, segment_dict


def preprocess_spectrum(spec_obs, temp_back):
    """Preprocess spectrum

    Args:
        spectrum (array): (N, 2). The first colunm gives the frequency and
            the second column gives the temperature.
        temp_back (float): Background temperature.
    """
    if spec_obs[0, 0] > spec_obs[-1, 0]: # Make freq ascending
        spec_obs = spec_obs[::-1]
    spec_obs[:, 1] = np.maximum(spec_obs[:, 1], temp_back)
    return spec_obs