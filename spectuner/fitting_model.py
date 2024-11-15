import numpy as np

from .preprocess import get_T_data
from .xclass_wrapper import derive_freq_range
from .identify import PeakManager


def create_fitting_model(obs_data, mol_store, config, T_base_data):
    param_mgr = mol_store.create_parameter_manager(config)
    # TODO: better way to create bounds?
    bounds = param_mgr.derive_bounds(config["opt"]["bounds"])
    model = FittingModel(
        obs_data, mol_store, bounds, config, T_base_data=T_base_data
    )
    return model


class FittingModel:
    def __init__(self, obs_data, mol_store, bounds, config, T_base_data=None):
        self.mol_store = mol_store
        self.include_list = mol_store.include_list
        self.sl_model = mol_store.create_spectral_line_model(config)
        T_back = self.sl_model.pm.T_back
        self.freq_range_data, self.freq_data, self.T_obs_data \
            = self._preprocess_spectra(obs_data)
        self.bounds = bounds
        #
        loss_fn = config.get("loss_fn", "pm")
        if loss_fn == "pm":
            self.loss_fn = PeakManager(
                obs_data, T_back, **config["peak_manager"],
                T_base_data=T_base_data
            )
        elif loss_fn == "mse":
            self.loss_fn = MSE(obs_data, T_back, T_base_data)
        else:
            raise ValueError(f"Unknown loss function {loss_fn}.")
        self.T_base_data = T_base_data
        self.T_back = T_back
        self.blob = config["opt"].get("blob", False)

    def _preprocess_spectra(self, obs_data):
        if isinstance(obs_data, list) or isinstance(obs_data, tuple):
            freq_range_data = []
            freq_data = []
            T_obs_data = []
            for spec in obs_data:
                freq_range_data.append(derive_freq_range(spec[:, 0]))
                freq_data.append(spec[:, 0])
                T_obs_data.append(spec[:, 1])
        elif isinstance(obs_data, np.ndarray):
            freq_range_data = [derive_freq_range(obs_data[:, 0])]
            freq_data = [obs_data[:, 0]]
            T_obs_data = [obs_data[:, 1]]
        else:
            raise ValueError("obs_data should be list, tuple or numpy array.")
        return freq_range_data, freq_data, T_obs_data

    def __call__(self, params):
        iterator = self.sl_model.call_multi(
            self.freq_data, self.include_list, params, remove_dir=True
        )
        T_pred_data = []
        for args in iterator:
            T_pred = args[0]
            T_pred = np.maximum(T_pred, self.T_back)
            T_pred_data.append(T_pred)
        loss, loss_pm = self.loss_fn(T_pred_data)
        if self.blob:
            return loss, loss_pm
        return loss

    def call_func(self, params, remove_dir=True):
        T_pred_data = []
        trans_data = []
        tau_data = []
        job_dir_data = []
        iterator = self.sl_model.call_multi(
            self.freq_data, self.include_list, params, remove_dir=remove_dir
        )
        for T_pred, _, trans, tau, job_dir in iterator:
            T_pred_data.append(T_pred)
            trans_data.append(trans)
            tau_data.append(tau)
            job_dir_data.append(job_dir)
        if len(T_pred_data) == 1:
            return T_pred_data[0], trans_data[0], tau_data[0], job_dir_data[0]
        return T_pred_data, trans_data, tau_data, job_dir_data


class MSE:
    def __init__(self, obs_data, T_back=0., T_base_data=None):
        T_obs_data = get_T_data(obs_data)
        if T_base_data is None:
            self.T_obs_data = T_obs_data
        else:
            self.T_obs_data = [T_obs - T_base + T_back for T_obs, T_base
                               in zip(T_obs_data, T_base_data)]

    def __call__(self, T_pred_data):
        loss = 0.
        for T_obs, T_pred in zip(self.T_obs_data, T_pred_data):
            loss += np.mean(np.square(T_obs - T_pred))
        loss /= len(T_pred_data)
        return loss, 0.