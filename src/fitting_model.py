import numpy as np

from .xclass_wrapper import create_wrapper_from_config, derive_freq_range


def create_fitting_model(spec_obs, mol_names, bounds,
                         config, vLSR=None, tBack=None):
    kwargs = {}
    if vLSR is not None:
        kwargs["vLSR"] = vLSR
    if tBack is not None:
        kwargs["tBack"] = tBack
    wrapper = create_wrapper_from_config(spec_obs, mol_names, config["xclass"], **kwargs)

    scaler = Scaler(wrapper.n_mol_param, wrapper.n_param_per_mol)
    bounds_mol = scaler.derive_bounds(bounds)

    bounds_misc = []
    bounds_dict = config["bounds"]
    for key in wrapper.params_misc:
        bounds_misc.append(bounds_dict[key])
    bounds_misc = np.vstack(bounds_misc)

    bounds = np.vstack([bounds_mol, bounds_misc])

    if config["loss_fn"] == "l1":
        loss_fn = l1_loss
    elif config["loss_fn"] == "l2":
        loss_fn = l2_loss
    else:
        raise ValueError("Unknown loss function.")
    return FittingModel(wrapper, bounds, scaler, spec_obs[:, 1], loss_fn)


def create_fitting_model_extra(obs_data, mol_dict, include_list,
                               config_xclass, config_opt,
                               vLSR=None, tBack=None, loss_fn=None):
    kwargs = {}
    if vLSR is not None:
        kwargs["vLSR"] = vLSR
    if tBack is not None:
        kwargs["tBack"] = tBack
    wrapper = create_wrapper_from_config(obs_data, mol_dict, config_xclass, **kwargs)

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
        kwargs_reg=config_opt.get("kwargs_reg", None)
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


class ScalerExtra:
    def __init__(self, pm):
        self.pm = pm

    def call(self, params):
        params_mol, params_iso, params_misc = self.split_params(params)
        params_mol = params_mol.copy()
        params_mol[:, :4] = 10**params_mol[:, :4]
        params_iso = 10**params_iso
        params_new = np.concatenate([np.ravel(params_mol), params_iso, params_misc])
        return params_new

    def derive_bounds(self, bounds_mol, bounds_iso, bounds_misc):
        # bounds_mol (5, 2)
        # bounds_iso (2,)
        # bounds_misc (dict)
        bounds_mol = np.tile(bounds_mol, (self.pm.n_mol, 1))
        bounds_iso = np.tile(bounds_iso, (self.pm.n_iso_param, 1))

        bounds_misc_ = []
        for key in self.pm.misc_names:
            bounds_misc_.append(bounds_misc[key])
        if len(bounds_misc_) == 0:
            bounds_misc = np.zeros((0, 2))
        else:
            bounds_misc = np.vstack(bounds_misc_)

        bounds = np.vstack([bounds_mol, bounds_iso, bounds_misc])
        return bounds

    def split_params(self, params):
        params_mol = params[self.pm.inds_mol_param]
        params_mol = params_mol.reshape(-1, self.pm.n_param_per_mol)
        params_iso = params[self.pm.inds_iso_param]
        params_misc = params[self.pm.inds_misc_param]
        return params_mol, params_iso, params_misc


class FittingModel:
    def __init__(self, obs_data, func, include_list, bounds, scaler, loss_fn, kwargs_reg=None):
        self.freq_range_data, self.freq_data, self.T_obs_data \
            = self._preprocess_spectra(obs_data)
        self.include_list = include_list
        self.func = func
        self.bounds = bounds
        self.scaler = scaler
        self.loss_fn = loss_fn
        if kwargs_reg is None:
            kwargs_reg = {}
        self.regularizer = ThresholdRegularizer(obs_data, **kwargs_reg)

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
        n_segment = len(self.T_obs_data)
        loss = [None]*n_segment
        T_pred_max = [None]*n_segment
        for i_segment in range(n_segment):
            T_obs = self.T_obs_data[i_segment]
            T_pred = self._call_single(i_segment, params, remove_dir=True)[0]
            loss[i_segment] = self.loss_fn(T_pred, T_obs)
            T_pred_max[i_segment] = T_pred.max()
        loss = np.mean(loss)
        loss += self.regularizer(max(T_pred_max))
        return loss

    def _call_single(self, i_segment, params, remove_dir):
        T_obs = self.T_obs_data[i_segment]
        self.func.update_frequency(*self.freq_range_data[i_segment])
        self.func.update_include_list(self.include_list[i_segment])
        params = self.derive_params(params)
        T_pred, _, trans, _, job_dir = self.func.call(params, remove_dir=remove_dir)
        # TODO Check this in the wrapper
        if T_pred is None:
            T_pred = np.zeros_like(T_obs)
        return T_pred, trans, job_dir

    def call_func(self, params, remove_dir=True):
        T_pred_data = []
        trans_data = []
        job_dir_data = []
        for i_segment in range(len(self.T_obs_data)):
            T_pred, trans, job_dir = self._call_single(i_segment, params, remove_dir)
            T_pred_data.append(T_pred)
            trans_data.append(trans)
            job_dir_data.append(job_dir)
        if len(T_pred_data) == 1:
            return T_pred_data[0], trans_data[0], job_dir_data[0]
        return T_pred_data, trans_data, job_dir_data

    def derive_params(self, params):
        if self.scaler is not None:
            params = self.scaler.call(params)
        return params


class ThresholdRegularizer:
    def __init__(self, obs_data, frac_cut=.25, alpha=0.01):
        T_obs = np.concatenate([spec[:, 1] for spec in obs_data])
        T_max = T_obs.max()
        T_median = np.median(T_obs)
        self.T_thr = T_median + frac_cut*(T_max - T_median)
        self.alpha = alpha

    def __call__(self, T_pred_max):
        return np.maximum(self.alpha*(self.T_thr - T_pred_max), 0.)