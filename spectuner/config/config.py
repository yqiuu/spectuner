from __future__ import annotations
import yaml
import shutil
import pickle
from typing import Optional, Literal, Tuple
from pprint import pformat
from copy import deepcopy
from pathlib import Path

import numpy as np

from ..sl_model import ParameterManager


__all__ = [
    "create_config",
    "load_preprocess_config",
    "load_config",
    "load_default_config",
    "save_config",
    "preprocess_config",
    "append_exclude_info",
    "Config",
]


def create_config(dir="./"):
    template_dir = Path(__file__).resolve().parent/Path("templates")
    target_dir = Path(dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    lines = open(template_dir/"config.yml").readlines()
    open(target_dir/"config.yml", "w").writelines(lines)

    for _, fname in iter_config_names():
        shutil.copy(template_dir/fname, target_dir)


def load_preprocess_config(dirname):
    config = load_config(dirname)
    preprocess_config(config)
    return config


def load_config(fname: str) -> Config:
    """Load the config.

    Args:
        fname: Path to the config file. If this is a directory, it will load
        the config defined by YAML files.

    Returns:
        Loaded config.
    """
    fname = Path(fname)
    if fname.is_file():
        config = pickle.load(open(fname, "rb"))
    elif fname.is_dir():
        config = yaml.safe_load(open(fname/"config.yml"))
        for key, name in iter_config_names():
            config[key] = yaml.safe_load(open(fname/name))
    else:
        raise ValueError(f"Unknown path: {fname}")
    return Config(**config)


def load_default_config() -> Config:
    """Load the default config.

    Returns:
        Default config.
    """
    return load_config(Path(__file__).parent/"templates")


def save_config(config: Config, fname: str):
    """Save the config to a pickle file.

    Args:
        config: Config to save.
        fname: Saving name.
    """
    if isinstance(config, Config):
        pickle.dump(dict(config), open(fname, "wb"))
    else:
        raise ValueError(f"Unknown input: {config}")


def preprocess_config(config):
    """This function does the following (in-place):
        1. Load ``spec`` in ``obs_info`` if applicable.
        2. Load ``freqs_exclude`` in ``peak_manager`` if applicable.
    """
    if config["obs_info"] is not None:
        for item in config["obs_info"]:
            if isinstance(item["spec"], str) or isinstance(item["spec"], Path):
                item["spec"] = np.loadtxt(item["spec"])
    freqs_exclude_in = config["peak_manager"]["freqs_exclude"]
    if isinstance(freqs_exclude_in, str) or isinstance(freqs_exclude_in, Path):
        config["peak_manager"]["freqs_exclude"] = np.loadtxt(freqs_exclude_in)

def iter_config_names():
    keys = ["species", "modify"]
    file_names = [
        "species.yml",
        "modify.yml"
    ]
    return zip(keys, file_names)


def append_exclude_info(config, freqs_exclude, exlude_list):
    config = deepcopy(config)
    if config["peak_manager"]["freqs_exclude"] is None:
        config["peak_manager"]["freqs_exclude"] = np.zeros(0)
    config["peak_manager"]["freqs_exclude"] \
        = np.append(config["peak_manager"]["freqs_exclude"], freqs_exclude)
    if config["species"]["exclude_list"] is None:
        config["species"]["exclude_list"] = []
    config["species"]["exclude_list"].extend(exlude_list)
    return config


class Config(dict):
    """A subclass of dict with user-friendly methods to update the config."""
    def __repr__(self) -> str:
        return pformat(dict(self), sort_dicts=False)

    def append_spectral_window(self,
                               spec: np.ndarray,
                               beam_info: float | tuple,
                               noise: float,
                               T_bg: float=0.,
                               need_cmb: bool=True):
        """Add a spectral window to ``obs_info``.

        Args:
            spec: The observed spectrum given by a 2D array, with the first
                column being the frequency in MHz and the second column
                being the intensity in K.
            beam_info: For single disk telescopes, this should
                be a float indicating the telescope diameter in meter. For
                interferometers, this should be (``BMAJ``, ``BMIN``) indicating
                the beam size in degree.
            noise: RMS noise in K.
            T_bg: Background temperature in K.
            need_cmb: If ``true``, additionally add 2.726 K to the background
                temperature.
        """
        if self["obs_info"] is None:
            self["obs_info"] = []

        item = {
            "spec": spec,
            "beam_info": beam_info,
            "T_bg": T_bg,
            "need_cmb": need_cmb,
            "noise": noise,
        }
        if len(self["obs_info"]) == 0:
            self["obs_info"].append(item)
            return

        spec_prev = self["obs_info"][-1]["spec"]
        freq_max_prev = np.max(spec_prev[:, 0])
        freq_min = np.min(item["spec"][:, 0])
        if freq_max_prev >= freq_min:
            raise ValueError("Spectral windows should be non-overlapping and in"
                             " asceding of frequency.")
        self["obs_info"].append(item)

    def set_fname_db(self, fname_db: str):
        """Set the path to the spectroscopic database.

        Args:
            fname_db: Path to the spectroscopic database.
        """
        self["sl_model"]["fname_db"] = fname_db

    def set_n_process(self, n_process: int):
        """Set the number of processes for the multiprocessing pool.

        Args:
            n_process: Number of processes.
        """
        self["n_process"] = n_process

    def set_peak_manager(self,
                         noise_factor: float=4.,
                         rel_height: float=0.25,
                         freqs_exclude: Optional[np.ndarray]=None):
        config_peak_mgr = self["peak_manager"]
        config_peak_mgr["noise_factor"] = noise_factor
        config_peak_mgr["rel_height"] = rel_height
        config_peak_mgr["freqs_exclude"] = freqs_exclude

    def set_param_info(self,
                       param_name: Literal["theta", "T_ex", "N_tot", "delta_v", "v_offset"],
                       is_log: bool,
                       bound: Tuple[float, float],
                       is_shared: bool=False,
                       special: Optional[str]=None):
        r"""Update the settings of a parameter.

        Args:
            param_name: Parameter name.
            is_log: Whether to use log-scale during fitting for this parameter.
            bound: Lower and upper limits used by optimizers for fitting. If
                ``is_log=True``, the limits should be in log-scale. For
                example, if ``is_log=True``, ``bound=(12, 20)`` means that
                the parameter is between :math:`10^{12}` and :math:`10^{20}`.
                The unit of the limits follows:

                -  theta: arcsec
                -  T_ex: K
                -  N_tot: cm^-2
                -  delta_v: km/s
                -  v_offset: km/s

            is_shared: Whether the parameter is shared in joint fitting of
                different states and isotopologues.
            special: Special parametrization. Now, this can only be used for
                ``theta``. If set ``special="eta"``, ``theta`` should be
                treated as the filling factor :math:`\eta`, with
                :math:`\eta = \theta^2/(\theta^2 + \theta_{\rm maj} \theta_{\rm min})`.
        """
        if param_name not in ParameterManager.param_names:
            raise ValueError("param_name should be one of {}.".format(
                ParameterManager.param_names))

        item = {
            "is_shared": is_shared,
            "is_log": is_log,
            "bound": bound,
        }
        if special is not None:
            item["special"] = special
        self["param_info"][param_name] = item

    def set_optimizer(self, method: str, **kwargs):
        """Set the optimizer for the fitting.

        Args:
            method (str): Name of the optimizer.

                - Use 'pso' (particle swarm optimization) for line
                  identification.
                - Use 'slsqp' (sequential least squares programming implemented
                  in ``scipy.minimize``), for pixel-by-pixel fitting with a
                  nerual network.

            **kwargs: Additional arguments for the optimizer.

                - n_trial (int): Number of trials for 'pso'. The optimizer will
                  be run for n_trial times, and the best fit will be selected
                  among all trials. Defaults to 1.
                - n_swarm (int): Number of particles for 'pso'. Defaults to 28.
                - n_draw (int): Number of samples drawed by the neural network.
                  This only works for local optimizers such as 'slsqp'. The
                  code will compute the fitness for each sample and select the
                  best one as the initial guess. Defaults to 50.
        """
        config_opt = self["optimizer"]
        config_opt["method"] = method
        config_opt.update(kwargs)

    def set_ident_species(self,
                          species: Optional[list],
                          collect_iso: bool=True,
                          combine_iso: bool=False,
                          combine_state: bool=False,
                          version_mode: Literal["default", "all"]="default",
                          include_hyper: bool=False,
                          exclude_list: Optional[list]=None,
                          rename_dict: Optional[dict]=None):
        """Set species for line identification.

        Args:
            species: List of species to include. If ``None``, inlcude all
                possible species in the given frequency ranges.
            collect_iso: If ``True``, collect isotopologues of molecules in
                ``species``.
            combine_iso: If ``True``, combine isotopologues and fitting them
                jointly.
            combine_state: If ``True``, combine states and fitting them jointly.
            version_mode: If set to ``'all'``, include all versions of the species
                in the indiviudal fitting phase. Then, during the combining phase,
                the best fit among the versions is used. Both ``combine_iso`` and
                ``combine_state`` must be ``False`` for this option to work.
                Defaults to ``'default'``.
            include_hyper: If ``True``, include hyperfine states.
            exclude_list: List of species to exclude.
            rename_dict: A dict to rename species.
        """
        config_species = self["species"]
        config_species["species"] = species
        config_species["collect_iso"] = collect_iso
        config_species["combine_iso"] = combine_iso
        config_species["combine_state"] = combine_state
        config_species["version_mode"] = version_mode
        config_species["include_hyper"] = include_hyper
        config_species["exclude_list"] = exclude_list
        config_species["rename_dict"] = rename_dict

    def set_modificaiton_lists(self,
                               exclude_id_list: Optional[list]=None,
                               exclude_name_list: Optional[list]=None,
                               include_id_list: Optional[list]=None):
        """Set modification lists.

        Args:
            exclude_id_list: List of IDs to be removed from the combined result.
            exclude_name_list: List of molecule names to be removed from the
                combined result.
            include_id_list: List of IDs to be added to the combined result.
        """
        if exclude_id_list is None:
            exclude_id_list = []
        if exclude_name_list is None:
            exclude_name_list = []
        if include_id_list is None:
            include_id_list = []
        config_modify = self["modify"]
        config_modify["exclude_id_list"] = exclude_id_list
        config_modify["exclude_name_list"] = exclude_name_list
        config_modify["include_id_list"] = include_id_list

    def set_pixel_by_pixel_fitting(self,
                                   species: list,
                                   loss_fn: Literal["pm", "chi2"]="pm",
                                   need_spectra: bool=True):
        """Set pixel-by-pixel fitting.

        Args:
            species: List of species to fit. The species will be fit jointly.
            need_spectra: Whether to save the best-fitting model spectrum.
        """
        self["cube"]["species"] = species
        self["cube"]["loss_fn"] = loss_fn
        self["cube"]["need_spectra"] = need_spectra

    def set_inference_model(self,
                            ckpt: str,
                            device: str="cuda:0",
                            batch_size: int=64,
                            num_workers: int=2):
        """Set AI model related parameters.

        Args:
            ckpt: Path to the checkpoint file.
            device: Device to use. Set ``device="cpu"`` if no GPU is available.
                Defaults to "cuda:0".
            batch_size: Batch size of inference. Reduce this number if GPU
                memory is not enough. Defaults to 64.
            num_workers: Number of workers for the data loader.
        """
        config_inf = self["inference"]
        config_inf["ckpt"] = ckpt
        config_inf["device"] = device
        config_inf["batch_size"] = batch_size
        config_inf["num_workers"] = num_workers