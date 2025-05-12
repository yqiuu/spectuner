import numpy as np


def load_preprocess(obs_info, clip=True):
    """Load observed spectra.

    The following preprocessing is performed:
        1. Ensure the frequency is ascending.
        2. Reset the intensity below zero to zero.

    Args:
        obs_info (list): A list of dicts that specify the observation
            information.

    Returns:
        list: Each element specifies a spectrum. The first colunm gives the
            frequency (MHz) and the second column gives the intensity (K).
    """
    obs_data = []
    for item in obs_info:
        if "spec" in item:
            obs_data.append(np.copy(item["spec"]))
    return [preprocess_spectrum(spec, clip) for spec in obs_data]


def preprocess_spectrum(spectrum, clip=True):
    """Preprocess spectrum

    The following preprocessing is performed:
        1. Ensure the frequency is ascending.
        2. Reset the intensity below zero to zero.

    Args:
        spectrum (array): (N, 2). The first colunm gives the frequency and
            the second column gives the temperature.
    """
    if spectrum[0, 0] > spectrum[-1, 0]: # Make freq ascending
        spectrum = spectrum[::-1]
    if clip:
        spectrum[:, 1] = np.maximum(spectrum[:, 1], 0.)
    return spectrum


def get_freq_data(data):
    return [spec[:, 0] for spec in data]

def get_T_data(data):
    return [spec[:, 1] for spec in data]