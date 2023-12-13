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


def l1_loss(y_pred, y_obs):
    return np.sum(np.abs(y_pred - y_obs))


def l2_loss(y_pred, y_obs):
    # y_pred, (N,)
    # y_obs, (N,)
    return np.sum(np.square(y_pred - y_obs))


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
