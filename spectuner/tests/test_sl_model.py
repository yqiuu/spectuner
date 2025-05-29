import pickle
from pathlib import Path

import numpy as np
from numpy import testing
import spectuner


def test_sl_model():
    data = pickle.load(open(get_fname(), "rb"))
    sl_data_list, freq_list, obs_info, params_list = create_sl_model()
    for params, spec_list_test in zip(params_list, data):
        slm_state = spectuner.create_spectral_line_model_state(sl_data_list, freq_list, obs_info)
        spec_list = spectuner.compute_effective_spectra(slm_state, params, need_individual=False)
        for spec, spec_test in zip(spec_list, spec_list_test):
            testing.assert_allclose(spec_test, spec, rtol=1e-3)


def create_test_data():
    data = []
    sl_data_list, freq_list, obs_info, params_list = create_sl_model()
    for params in params_list:
        slm_state = spectuner.create_spectral_line_model_state(sl_data_list, freq_list, obs_info)
        data.append(spectuner.compute_effective_spectra(slm_state, params, need_individual=False))
    pickle.dump(data, open(get_fname(), "wb"))


def create_sl_model():
    freq_list = [np.linspace(50000, 54000, 2000), np.linspace(56000, 60000, 1000)]

    x_T = np.linspace(0, 200, 100)
    Q_T = 1e2*x_T*x_T
    N_t = 20

    obs_info = (
        {"beam_info": 65, "T_bg": 30., "need_cmb": True},
        {"beam_info": (1/3600, 1/3600), "T_bg": 30., "need_cmb": True},
    )

    sl_data_list = [
        {
            "freq": np.linspace(51000, 59000, N_t),
            "A_ul": np.linspace(1e-5, 3e-5, N_t),
            "E_low": np.linspace(50, 250, N_t),
            "g_u": np.linspace(10, 200, N_t),
            "x_T": x_T,
            "Q_T": Q_T,
        },
        {
            "freq": np.linspace(52000, 58000, N_t),
            "A_ul": np.linspace(2e-5, 3.5e-5, N_t),
            "E_low": np.linspace(50, 250, N_t),
            "g_u": np.linspace(10, 200, N_t),
            "x_T": x_T,
            "Q_T": Q_T,
        },
    ]

    for sl_data in sl_data_list:
        segment = np.zeros(len(sl_data["freq"]), dtype=int)
        for i_segment, freq in enumerate(freq_list):
            cond = (sl_data["freq"] >= freq[0]) & (sl_data["freq"] <= freq[-1])
            segment[cond] = i_segment
        sl_data["segment"] = segment

    params_list = [
        np.array([[.5, 100, 1e18, 20, -1.], [.5, 130, 1e18, 20, 1.5]]),
        np.array([[.5, 10, 1e17, 20, -2.], [.5, 12, 1e17, 20, -1.]]),
    ]

    return sl_data_list, freq_list, obs_info, params_list


def get_fname():
    return Path(__file__).parent/"data"/"test_sl_model.pickle"


if __name__ == "__main__":
    create_test_data()