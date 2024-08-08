from copy import deepcopy

import numpy as np


def load_preprocess(obs_data, T_back):
    """Load observed spectra.

    The function resets the intensity below ``T_back`` to ``T_back``.

    Args:
        obs_data (str or list): Data of observed spectra.
            - str: Load the text file specified by the string.
            - list of str: Load every text file specified by the string.
            - list of array: Each array should be a spectrum. The first colunm
            gives the frequency (MHz) and the second column gives the intensity
            (K).
        T_back (float): Background temperature.

    Returns:
        list: Each element specifies a spectrum. The first colunm gives the
            frequency (MHz) and the second column gives the intensity (K).
    """
    if isinstance(obs_data, str):
        obs_data = [obs_data]

    obs_data_ret = []
    for item in obs_data:
        if isinstance(item, str):
            obs_data_ret.append(np.loadtxt(item))
        else:
            obs_data_ret.append(deepcopy(item))

    obs_data_ret = [preprocess_spectrum(spec, T_back) for spec in obs_data_ret]
    return obs_data_ret


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