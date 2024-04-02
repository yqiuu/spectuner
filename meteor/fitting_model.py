import numpy as np

from .xclass_wrapper import derive_freq_range
from .algorithms import derive_median_frac_threshold
from .identify import PeakMatchingLoss, PeakManager


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


def create_fitting_model(obs_data, mol_store, config, config_opt, base_data):
    pm = mol_store.create_parameter_manager(config["sl_model"])
    # TODO: better way to create bounds?
    bounds = pm.scaler.derive_bounds(
        pm, config_opt["bounds_mol"], config_opt["bounds_iso"], {}
    )
    model = FittingModel(
        obs_data, mol_store, bounds, config, base_data=base_data
    )
    return model


class FittingModel:
    def __init__(self, obs_data, mol_store, bounds, config, base_data=None):
        self.mol_store = mol_store
        self.include_list = mol_store.include_list
        self.sl_model = mol_store.create_spectral_line_model(config["sl_model"])
        T_back = self.sl_model.pm.T_back
        self.freq_range_data, self.freq_data, self.T_obs_data \
            = self._preprocess_spectra(obs_data)
        self.bounds = bounds
        #
        loss_fn = config.get("loss_fn", "mae")
        if loss_fn == "mae":
            self.loss_fn = l1_loss
        elif loss_fn == "mse":
            self.loss_fn = l2_loss
        else:
            raise ValueError(f"Unknown loss function {loss_fn}.")
        #
        config_pm_loss = config.get("pm_loss", None)
        if config_pm_loss is None:
            self.pm_loss_fn = None
        else:
            self.pm_loss_fn = PeakManager(obs_data, T_back, **config_pm_loss)
        #
        config_thr_loss=config.get("thr_loss", None)
        if config_thr_loss is None:
            self.thr_loss_fn = None
        else:
            self.thr_loss_fn = ThresholdRegularizer(obs_data, **config_thr_loss)

        if base_data is None:
            base_data = [None for _ in range(len(obs_data))]
        else:
            base_data = [T_base - T_back for T_base in base_data]
        self.base_data = base_data
        self.T_back = T_back

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
                T_pred = np.full_like(T_obs, self.T_back)
            T_base = self.base_data[i_segment]
            if T_base is not None:
                T_pred = T_pred + T_base
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