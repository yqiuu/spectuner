import numpy as np

from .xclass_wrapper import derive_freq_range
from .algorithms import derive_median_frac_threshold, PeakMatchingLoss


def l1_loss(y_pred, y_obs):
    return np.mean(np.abs(y_pred - y_obs))


def l1_loss_log(y_obs, y_pred):
    delta = y_obs - y_pred
    return np.mean(np.log(1 + np.abs(delta)))


def l2_loss(y_pred, y_obs):
    # y_pred, (N,)
    # y_obs, (N,)
    return np.mean(np.square(y_pred - y_obs))


def l2_loss_log(y_obs, y_pred):
    delta = y_obs - y_pred
    return np.mean(np.log(1 + np.square(delta)))


class FittingModel:
    def __init__(self, obs_data, mol_store, bounds, config_slm,
                 config_pm_loss=None, config_thr_loss=None, base_data=None):
        self.mol_store = mol_store
        self.include_list = mol_store.include_list
        self.sl_model = mol_store.create_spectral_line_model(config_slm)
        T_back = self.sl_model.pm.T_back
        obs_data = self._remove_base(obs_data, base_data, T_back)
        self.freq_range_data, self.freq_data, self.T_obs_data \
            = self._preprocess_spectra(obs_data)
        self.bounds = bounds
        #
        self.loss_fn = l1_loss
        #
        if config_pm_loss is None:
            self.pm_loss_fn = None
        else:
            self.pm_loss_fn = PeakMatchingLoss(obs_data, T_back, **config_pm_loss)
        #
        if config_thr_loss is None:
            self.thr_loss_fn = None
        else:
            self.thr_loss_fn = ThresholdRegularizer(obs_data, **config_thr_loss)

    def _remove_base(self, obs_data, base_data, T_back):
        if base_data is None:
            return obs_data

        obs_data_new = []
        for spec, T_base in zip(obs_data, base_data):
            spec[:, 1] = spec[:, 1] - T_base + T_back
            obs_data_new.append(spec)
        return obs_data_new

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
        return freq_range_data, freq_data, T_obs_data,

    def __call__(self, params):
        loss = 0.
        T_pred_max = -np.inf
        n_segment = len(self.T_obs_data)
        iterator = self.sl_model.call_multi(
            self.freq_data, self.include_list, params, remove_dir=True
        )
        for i_segment, args in enumerate(iterator):
            T_obs = self.T_obs_data[i_segment]
            T_pred = args[0]
            if T_pred is None:
                T_pred = np.zeros_like(T_obs)
            loss += self.loss_fn(T_pred, T_obs)
            if self.pm_loss_fn is not None:
                loss += self.pm_loss_fn(i_segment, T_pred)
            T_pred_max = max(T_pred_max, T_pred.max())
        loss = loss/n_segment
        if self.thr_loss_fn is not None:
            loss += self.thr_loss_fn(T_pred_max)
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


class ThresholdRegularizer:
    def __init__(self, obs_data, alpha=1., T_thr=None, median_frac=.25):
        self.alpha = alpha
        if T_thr is None:
            T_thr = derive_median_frac_threshold(obs_data, median_frac)
        self.T_thr = T_thr

    def __call__(self, T_pred_max):
        return np.maximum(self.alpha*(self.T_thr - T_pred_max), 0.)