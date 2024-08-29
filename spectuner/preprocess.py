import numpy as np


def load_preprocess(obs_info):
    """Load observed spectra.

    1. Ensure the frequency is ascending.
    2. Resets the intensity below zero to zero.

    Args:
        obs_info (list): A list of dicts that specify the observation
            information.
            - If ``spec`` in the dict, load the corresponding array,
                with the first column being the frequency (MHz) and the second
                column being the intensity (K).
            - If ``fname`` in the dict, load the corresponding text file.

    Returns:
        list: Each element specifies a spectrum. The first colunm gives the
            frequency (MHz) and the second column gives the intensity (K).
    """
    obs_data = []
    for item in obs_info:
        if "spec" in item:
            obs_data.append(np.copy(item["spec"]))
        elif "fname" in item:
            obs_data.append(np.loadtxt(item["fname"]))
    return [preprocess_spectrum(spec) for spec in obs_data]


def preprocess_spectrum(spectrum):
    """Preprocess spectrum

    Args:
        spectrum (array): (N, 2). The first colunm gives the frequency and
            the second column gives the temperature.
    """
    if spectrum[0, 0] > spectrum[-1, 0]: # Make freq ascending
        spectrum = spectrum[::-1]
    spectrum[:, 1] = np.maximum(spectrum[:, 1], 0.)
    return spectrum


def get_freq_data(data):
    return [spec[:, 0] for spec in data]

def get_T_data(data):
    return [spec[:, 1] for spec in data]