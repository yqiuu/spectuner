from __future__ import annotations
from typing import Optional, Callable, Literal
from dataclasses import dataclass
from copy import deepcopy
from collections import defaultdict

import numpy as np

from .preprocess import load_preprocess, get_freq_data, get_T_data
from .sl_model import (
    create_spectral_line_model_state,
    create_spectral_line_db,
    SpectralLineDB,
    SpectralLineModel,
    ParameterManager,
)
from .peaks import PeakManager
from .utils import pick_default_kwargs


def combine_specie_lists(specie_lists, params_list):
    specie_list_ret = []
    for specie_list in specie_lists:
        specie_list_ret.extend(specie_list)
    params_ret = np.concatenate(params_list)
    return specie_list_ret, params_ret


def sum_T_single_data(T_single_dict, T_back=0., key=None):
    # Get a test dict
    for sub_dict in T_single_dict.values():
        for T_single_data in sub_dict.values():
            break
        break
    T_ret_data = [None for _ in T_single_data]

    def sum_sub(target_dict):
        for T_single_data in target_dict.values():
            for i_segment, T_single in enumerate(T_single_data):
                if T_single is None:
                    continue
                if T_ret_data[i_segment] is None:
                    T_ret_data[i_segment] = T_back
                T_ret_data[i_segment] = T_ret_data[i_segment] + T_single - T_back

    if key is not None:
        sum_sub(T_single_dict[key])
        return T_ret_data

    for sub_dict in T_single_dict.values():
        sum_sub(sub_dict)
    return T_ret_data


def compute_T_single_data(slm_factory: SpectralLineModelFactory,
                          obs_info: list,
                          specie_list: list,
                          params: np.ndarray) -> dict:
    T_single_data = defaultdict(dict)
    for item in specie_list:
        for name in item["species"]:
            specie_list_single, \
            params_single = derive_sub_specie_list_with_params(
                slm_factory, obs_info, specie_list, [name], params
            )
            sl_model = slm_factory.create_sl_model(obs_info, specie_list_single)
            T_single_data[item["id"]][name] = sl_model(params_single)
    T_single_data = dict(T_single_data)
    return T_single_data


def derive_sub_specie_list(specie_list, species):
    """Return a new specie list that only contains the given species.

    Args:
        specie_list (list): Specie list.
        species (list): A list of specie names that should be included.

    Returns:
        list: Filtered specie list.
    """
    species_list_new = []
    for item in specie_list:
        species_new = [name for name in item["species"] if name in species]
        if len(species_new) > 0:
            item_new = deepcopy(item)
            item_new["species"] = species_new
            species_list_new.append(item_new)
    return species_list_new


def derive_sub_specie_list_with_params(slm_factory, obs_info, specie_list, species, params):
    """Extract a sub specie list and corresponding parameters.

    Args:
        specie_list (list): Specie list.
        species (list): A list of specie names that should be included.
        params (array): Parameters.
        config (dict): Config

    Returns:
        list: Filtered specie list.
        array: Filtered parameters.
    """
    specie_list_sub = derive_sub_specie_list(specie_list, species)
    param_mgr = slm_factory.create_parameter_mgr(specie_list, obs_info)
    params_sub = param_mgr.get_subset_params(species, params)
    return specie_list_sub, params_sub


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
            self._sl_db = create_spectral_line_db(config["sl_model"]["fname_db"])
        else:
            self._sl_db = sl_db

    def create_parameter_mgr(self, specie_list: list, obs_info: list):
        param_info = self._config["sl_model"]["params"]
        return ParameterManager(specie_list, param_info, obs_info)

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
        kwargs = pick_default_kwargs(
            create_spectral_line_model_state, self._config["sl_model"]
        )
        slm_state = create_spectral_line_model_state(
            sl_data_list=sl_dict_list_,
            freq_list=freq_data,
            obs_info=obs_info,
            **kwargs,
        )
        param_mgr = self.create_parameter_mgr(specie_list, obs_info)
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
                             loss_fn: Literal["pm", "chi2"]="pm",
                             sl_dict_list: Optional[list]=None,
                             T_base_data: Optional[list]=None) -> FittingModel:
        sl_model = self.create_sl_model(obs_info, specie_list, sl_dict_list)
        # TODO: allow to have different loss functions
        if loss_fn == "pm":
            loss_fn = self.create_peak_mgr(obs_info, T_base_data)
        elif loss_fn == "chi2":
            obs_data = load_preprocess(obs_info)
            loss_fn = ChiSquare(obs_data, T_base_data)
        else:
            raise ValueError(f"Unknown fitting loss {loss_fn}.")
        bounds = sl_model.param_mgr.derive_bounds(self._config["bound_info"])
        return FittingModel(obs_info, sl_model, loss_fn, bounds)


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


class ChiSquare:
    def __init__(self, obs_data, T_base_data=None):
        T_obs_data = get_T_data(obs_data)
        if T_base_data is None:
            self.T_obs_data = T_obs_data
        else:
            self.T_obs_data = [T_obs - T_base for T_obs, T_base
                               in zip(T_obs_data, T_base_data)]

    def __call__(self, T_pred_data):
        loss = 0.
        for T_obs, T_pred in zip(self.T_obs_data, T_pred_data):
            loss += np.sum(np.square(T_obs - T_pred))
        loss *= .5
        return loss