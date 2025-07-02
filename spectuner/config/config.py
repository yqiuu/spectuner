import yaml
import shutil
from typing import Union, Optional, Literal
from copy import deepcopy
from pathlib import Path


import numpy as np


__all__ = [
    "create_config",
    "load_preprocess_config",
    "load_config",
    "load_default_config",
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


def load_config(dir):
    dir = Path(dir)
    config = yaml.safe_load(open(dir/"config.yml"))
    for key, fname in iter_config_names():
        config[key] = yaml.safe_load(open(dir/fname))
    return Config(**config)


def load_default_config():
    return load_config(Path(__file__).parent/"templates")


def preprocess_config(config):
    """This function does the following (in-place):
        1. Load ``spec`` in ``obs_info`` if applicable.
        2. Derive and set ``prominence`` in ``config["peak_manager"]``.
        3. Load ``freqs_exclude`` in ``peak_manager`` if applicable.
    """
    if config["obs_info"] is not None:
        for item in config["obs_info"]:
            if not isinstance(item["spec"], np.ndarray):
                item["spec"] = np.loadtxt(item["spec"])

    if "prominence" not in config["peak_manager"] \
        and config["obs_info"] is not None:
        noises = np.array([item["noise"] for item in config["obs_info"]])
        noise_factor = config["peak_manager"].pop("noise_factor")
        config["peak_manager"]["prominence"] = noise_factor*noises

    if not isinstance(config["peak_manager"]["prominence"], np.ndarray):
        config["peak_manager"]["freqs_exclude"] \
            = np.loadtxt(config["peak_manager"]["freqs_exclude"])


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
            beam_info (float | tuple): For single disk telescopes, this should
                be a float indicating the telescope diameter in meter. For
                interferometers, this should be (``BMAJ``, ``BMIN``) indicating
                the beam size in degree.
            noise (float): RMS noise in K.
            T_bg (float, optional): Background temperature in K.
            need_cmb (bool, optional): If ``true``, additionally add 2.726 K to
                the background temperature.
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

    def set_n_process(self, n_process):
        """Set the number of processes for the multiprocessing pool.

        Args:
            n_process: Number of processes.
        """
        self["n_process"] = n_process

    def set_peak_manager(self,
                         noise_factor: float=4.,
                         rel_height: float=0.25,
                         freqs_exclude: Union[str, np.ndarray, None]=None):
        config_peak_mgr = self["peak_manager"]
        config_peak_mgr["noise_factor"] = noise_factor
        config_peak_mgr["rel_height"] = rel_height
        config_peak_mgr["freqs_exclude"] = freqs_exclude

    def set_ident_species(self,
                          speices: Optional[list],
                          collect_iso: bool=True,
                          combine_iso: bool=False,
                          combine_state: bool=False,
                          include_hyper: bool=False,
                          exclude_list: Optional[list]=None,
                          rename_dict: Optional[dict]=None):
        """Set species for line identification.

        Args:
            speices: List of species to include. If ``None``, inlcude all
                possible species in the given frequency ranges.
            collect_iso: If ``True``, collect isotopologues of molecules in
                ``species``.
            combine_iso: If ``True``, combine isotopologues and fitting them
                jointly.
            combine_state: If ``True``, combine states and fitting them jointly.
            include_hyper: If ``True``, include hyperfine states.
            exclude_list: List of species to exclude.
            rename_dict: A dict to rename species.
        """
        config_species = self["species"]
        config_species["species"] = speices
        config_species["collect_iso"] = collect_iso
        config_species["combine_iso"] = combine_iso
        config_species["combine_state"] = combine_state
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