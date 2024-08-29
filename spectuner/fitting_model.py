import numpy as np

from .preprocess import get_T_data
from .xclass_wrapper import derive_freq_range
from .peaks import PeakManager


def jit_fitting_model(model):
    """Call the fitting model once to enable jit."""
    model(np.mean(model.bounds, axis=1))


class FittingModel:
    def __init__(self, slm_factory, specie_list, obs_data, bounds_info, loss_fn,
                 T_base_data=None, blob=False):
        self.sl_model = slm_factory.create(specie_list)
        self.bounds = self.sl_model.param_mgr.derive_bounds(bounds_info)
        self.freq_range_data, self.freq_data, self.T_obs_data \
            = self._preprocess_spectra(obs_data)
        #
        self.loss_fn = loss_fn
        self.T_base_data = T_base_data
        self.blob = blob

    @classmethod
    def from_config(cls, slm_factory, specie_list, obs_data, config, T_base_data=None):
        loss_fn_type = config.get("loss_fn", "pm")
        if loss_fn_type == "pm":
            loss_fn = PeakManager(
                obs_data, **config["peak_manager"], T_base_data=T_base_data
            )
        elif loss_fn_type == "mse":
            loss_fn = MSE(obs_data)
        else:
            raise ValueError(f"Unknown loss function {loss_fn}.")
        return cls(
            slm_factory, specie_list, obs_data,
            bounds_info=config["opt"]["bounds"],
            loss_fn=loss_fn,
            T_base_data=T_base_data
        )

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
        T_pred_list = self.sl_model(params)
        loss, loss_pm = self.loss_fn(T_pred_list)
        if self.blob:
            return loss, loss_pm
        return loss


class MSE:
    def __init__(self, obs_data):
        self.T_obs_data = get_T_data(obs_data)

    def __call__(self, T_pred_data):
        loss = 0.
        for T_obs, T_pred in zip(self.T_obs_data, T_pred_data):
            loss += np.mean(np.square(T_obs - T_pred))
        loss /= len(T_pred_data)
        return loss