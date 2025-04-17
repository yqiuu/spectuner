from __future__ import annotations
from typing import Optional, Callable
from dataclasses import dataclass

import numpy as np

from .preprocess import load_preprocess, get_freq_data, get_T_data
from .sl_model import (
    create_spectral_line_model_state,
    SpectralLineDB,
    SQLSpectralLineDB,
    SpectralLineModel,
    ParameterManager,
)
from .peaks import PeakManager


def jit_fitting_model(model):
    """Call the fitting model once to enable jit."""
    model(np.mean(model.bounds, axis=1))


class SpectralLineModelFactory:
    """Spectral line model factory.

    Args:
        config: Must have the following structure:

            sl_model:
                fname_db: str # Path to the spectroscopic database
                trunc: float
                eps_grid: float
                params: dict
                    theta:
                        is_log: bool
                        is_shared: bool
                        special: str
                    T_ex:
                        is_log: bool
                        is_shared: bool
                    N_tot:
                        is_log: bool
                        is_shared: bool
                    delta_v:
                        is_log: bool
                        is_shared: bool
                    v_LSR:
                        is_log: bool
                        is_shared: bool
            peak_manager:
                noise_factor: float
                rel_height: float
                freqs_exclude:
            bound_info: dict # Only used to create fitting models
                - theta: list (lower, upper)
                - T_ex: list (lower, upper)
                - N_tot: list (lower, upper)
                - delta_v: list (lower, upper)
                - v_LSR: list (lower, upper)
    """
    def __init__(self,
                 config: dict,
                 sl_db: Optional[SpectralLineDB]=None) -> None:
        self._config = config
        if sl_db is None:
            self._sl_db = SQLSpectralLineDB(config["sl_model"]["fname_db"])
        else:
            self._sl_db = sl_db

    def create_sl_model(self,
                        obs_info: list,
                        specie_list: list,
                        sl_dict_list: Optional[list]=None) -> SpectralLineModel:
        # Create sl_model
        obs_data = load_preprocess(obs_info)
        freq_data = get_freq_data(obs_data)
        if sl_dict_list is None:
            sl_dict_list_ = []
            for item in specie_list:
                for specie in item["species"]:
                    sl_dict_list_.append(
                        self._sl_db.query_sl_dict(specie, freq_data))
        else:
            sl_dict_list_ = sl_dict_list
        slm_state = create_spectral_line_model_state(
            sl_data_list=sl_dict_list_,
            freq_list=freq_data,
            obs_info=obs_info,
            trunc=self._config["sl_model"]["trunc"],
            eps_grid=self._config["sl_model"]["eps_grid"],
        )
        param_info = self._config["sl_model"]["params"]
        param_mgr = ParameterManager(
            specie_list, param_info, obs_info
        )
        return SpectralLineModel(slm_state, param_mgr)

    def create_peak_mgr(self,
                        obs_info: list,
                        T_base_data: Optional[list]=None) -> PeakManager:
        obs_data = load_preprocess(obs_info)
        if "noise_factor" in self._config["peak_manager"]:
            noise_factor = self._config["peak_manager"]["noise_factor"]
            prominence = [noise_factor*item["noise"] for item in obs_info]
        else:
            prominence = self._config["peak_manager"]["prominence"]
        return PeakManager(
            obs_data,
            prominence=prominence,
            rel_height=self._config["peak_manager"]["rel_height"],
            freqs_exclude=self._config["peak_manager"]["freqs_exclude"],
            T_base_data=T_base_data
        )

    def create_fitting_model(self,
                             obs_info: list,
                             specie_list: list,
                             sl_dict_list: Optional[list]=None,
                             T_base_data: Optional[list]=None) -> FittingModel:
        sl_model = self.create_sl_model(obs_info, specie_list, sl_dict_list)
        # TODO: allow to have different loss functions
        peak_mgr = self.create_peak_mgr(obs_info, T_base_data)
        bounds = sl_model.param_mgr.derive_bounds(self._config["bound_info"])
        return FittingModel(obs_info, sl_model, peak_mgr, bounds)


@dataclass(frozen=True)
class FittingModel:
    obs_info: list
    sl_model: SpectralLineModel
    loss_fn: Callable
    bounds: np.ndarray
    blob: bool = False

    def __call__(self, params):
        T_pred_data = self.sl_model(params)
        value = self.loss_fn(T_pred_data)
        if self.blob:
            if isinstance(value, tuple):
                return value
            return value, None
        if isinstance(value, tuple):
            return value[0]
        return value


class MSE:
    def __init__(self, obs_data):
        self.T_obs_data = get_T_data(obs_data)

    def __call__(self, T_pred_data):
        loss = 0.
        for T_obs, T_pred in zip(self.T_obs_data, T_pred_data):
            loss += np.mean(np.square(T_obs - T_pred))
        loss /= len(T_pred_data)
        return loss