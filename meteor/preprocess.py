import glob

import numpy as np


def load_preprocess(file_spec, T_back):
    if isinstance(file_spec, str) and file_spec.startswith("glob:"):
        file_spec = glob.glob(file_spec.replace("glob:", ""))
    elif isinstance(file_spec, str):
        file_spec = [file_spec]

    obs_data = [np.loadtxt(fname) for fname in file_spec]
    obs_data = [preprocess_spectrum(spec, T_back) for spec in obs_data]
    return obs_data


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


def get_freq_data(data):
    return [spec[:, 0] for spec in data]

def get_T_data(data):
    return [spec[:, 1] for spec in data]