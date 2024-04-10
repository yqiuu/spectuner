import pickle
import math
from pathlib import Path

import numpy as np
import meteor


def test_pm_loss():
    data = pickle.load(open(get_fname(), "rb"))
    peak_mgr = create_peak_manager(data["obs_data"])
    loss = peak_mgr(0, data["y_pred"])
    assert math.isclose(loss, data["loss"], rel_tol=1e-5)


def create_test_data():
    def gaussian_profile(x, amp, mu, std):
        delta = np.square((x - mu)/std)
        return amp*np.exp(-delta*delta)

    x = np.linspace(-1., 1., 1000)
    y_obs = gaussian_profile(x, 1., -.7, .05) \
        + gaussian_profile(x, .7, .5, 0.1) \
        + gaussian_profile(x, .75, .35, .05)

    y_pred = gaussian_profile(x, 1., -.65, .05) \
        + gaussian_profile(x, .7, .45, 0.1) \
        + gaussian_profile(x, .3, 0., .07)
    obs_data = [np.vstack([x, y_obs]).T]

    peak_mgr = create_peak_manager(obs_data)
    loss = peak_mgr(0, y_pred)

    save_dict = {
        "obs_data": obs_data,
        "y_pred": y_pred,
        "loss": loss
    }
    pickle.dump(save_dict, open(get_fname(), "wb"))


def create_peak_manager(obs_data):
    T_back = 0.
    prominence = .1
    rel_height = .25
    return meteor.PeakManager(obs_data, T_back, prominence, rel_height)


def get_fname():
    return Path(__file__).parent/"data"/"mock_spectra.pickle"


if __name__ == "__main__":
    create_test_data()