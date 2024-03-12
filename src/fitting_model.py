from typing import Any
import numpy as np

from .xclass_wrapper import create_wrapper_from_config, derive_freq_range
from .algorithms import derive_median_frac_threshold, PeakMatchingLoss


def create_fitting_model(obs_data, mol_list, include_list,
                         config_xclass, config_opt,
                         vLSR=None, tBack=None, loss_fn=None,
                         base_data=None):
    kwargs = {}
    if vLSR is not None:
        kwargs["vLSR"] = vLSR
    if tBack is not None:
        kwargs["tBack"] = tBack
    wrapper = create_wrapper_from_config(obs_data, mol_list, config_xclass, **kwargs)

    scaler = ScalerExtra(wrapper.pm)
    bounds = scaler.derive_bounds(
        config_opt["bounds_mol"],
        config_opt["bounds_iso"],
        config_opt["bounds_misc"]
    )

    if loss_fn is not None:
        pass
    elif config_opt["loss_fn"] == "l1":
        loss_fn = l1_loss
    elif config_opt["loss_fn"] == "l1_log":
        loss_fn = l1_loss_log
    elif config_opt["loss_fn"] == "l2":
        loss_fn = l2_loss
    elif config_opt["loss_fn"] == "l2_log":
        loss_fn = l2_loss_log
    else:
        raise ValueError("Unknown loss function.")
    return FittingModel(
        obs_data, wrapper, include_list, bounds, scaler, loss_fn,
        **config_opt.get("kwargs", {}),
        base_data=base_data
    )


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


class Scaler:
    def __init__(self, n_mol_param, n_param_per_mol):
        self.n_mol_param = n_mol_param
        self.n_param_per_mol = n_param_per_mol

    def call(self, params):
        # Use log scale for the first three parameters.
        # source_size, T_rot, n_tot
        # params_mol (N_mol, N_params)
        params_mol, params_misc = self.split_params(params)
        params_mol = params_mol.copy()
        params_mol[:, :3] = 10**params_mol[:, :3]
        params_new = np.append(np.ravel(params_mol), params_misc)
        return params_new

    def derive_bounds(self, bounds):
        bounds = bounds.copy()
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        lb_mol, lb_misc = self.split_params(lb)
        lb_mol[:, :3] = np.log10(lb_mol[:, :3])
        lb_new = np.append(lb_mol, lb_misc)
        ub_mol, ub_misc = self.split_params(ub)
        ub_mol[:, :3] = np.log10(ub_mol[:, :3])
        ub_new = np.append(ub_mol, ub_misc)
        bounds_new = np.vstack([lb_new, ub_new]).T
        return bounds_new

    def split_params(self, params):
        # params (N,)
        params_mol = params[:self.n_mol_param]
        params_mol = params_mol.reshape(-1, self.n_param_per_mol)
        params_misc = params[self.n_mol_param:]
        return params_mol, params_misc


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