import numpy as np

from .xclass_wrapper import create_wrapper_from_config


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


def create_fitting_model_extra(spec_obs, mol_dict, config_xclass, config_opt,
                               vLSR=None, tBack=None, loss_fn=None):
    kwargs = {}
    if vLSR is not None:
        kwargs["vLSR"] = vLSR
    if tBack is not None:
        kwargs["tBack"] = tBack
    wrapper = create_wrapper_from_config(spec_obs, mol_dict, config_xclass, **kwargs)

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
    return FittingModel(wrapper, bounds, scaler, spec_obs[:, 1], loss_fn)


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
        params_mol[:, :3] = 10**params_mol[:, :3]
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
    def __init__(self, func, bounds, scaler, y_obs, loss_fn):
        self.func = func
        self.bounds = bounds
        self.scaler = scaler
        self.y_obs = y_obs
        self.loss_fn = loss_fn

    def __call__(self, params):
        params = self.derive_params(params)
        y_pred = self.func.call(params)
        # TODO Check this in the wrapper
        if y_pred is None:
            y_pred = np.zeros_like(self.y_obs)
        return self.loss_fn(y_pred, self.y_obs)

    def derive_params(self, params):
        if self.scaler is not None:
            params = self.scaler.call(params)
        return params
