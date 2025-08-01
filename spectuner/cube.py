from __future__ import annotations
import json
import multiprocessing as mp
from typing import Optional, Callable, Literal, Union
from functools import partial
from copy import copy
from pathlib import Path
from dataclasses import dataclass, asdict

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from astropy import units, constants
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from tqdm import tqdm

from . import ai
from .version import __version__
from .sl_model import (
    create_spectral_line_db,
    const_factor_mu_sigma,
    SQLSpectralLineDB
)
from .peaks import check_overlap
from .slm_factory import SpectralLineModelFactory
from .optimize import create_optimizer, Optimizer
from .preprocess import preprocess_spectrum


PARAM_NAMES = ("theta", "T_ex", "N_tot", "delta_v", "v_offset")
PARAM_UNITS = ("arcsec", "K", "cm-2", "km/s", "km/s")


def fit_pixel_by_pixel(config: dict,
                       fname_cube: str,
                       save_name: str,
                       sl_db: Optional[SQLSpectralLineDB]=None):
    """Fit the spectra of a cube.

    Args:
        config: The following config is required:

            - Set sl_model/fname_db to the path of the database.
            - Set bound_info.
            - Set species/species for species to be fitted.
            - Set species/combine_iso and species/combine_state to False.
            - Set optimizer/ for optimization configuration.
            - Set inference/ for AI model configuration.

        fname_cube: Path to the cube data.
        save_name: Path to the output file.
        sl_db: Input spectroscopic database. If ``None``, load the one specified
            in sl_model/fname_db.
    """
    if sl_db is None:
        sl_db = create_spectral_line_db(config["sl_model"]["fname_db"])

    species = config["cube"]["species"]

    # Load cube properties
    n_pixel = h5py.File(fname_cube)["index"].shape[0]
    freq_data = load_misc_data(fname_cube)[0]

    #
    if config["inference"]["ckpt"] is None:
        inf_model = None
        slm_factory = SpectralLineModelFactory(config, sl_db=sl_db)
    else:
        inf_model = ai.InferenceModel.from_config(config, sl_db=sl_db)
        slm_factory = inf_model.slm_factory
    opt = create_optimizer(config)
    need_spectra = config["cube"]["need_spectra"]
    postprocess = _AddExtraProps(slm_factory, opt, need_spectra=need_spectra)

    # Initialize saver
    parent_conn, child_conn = mp.Pipe()
    saver = mp.Process(
        target=_cube_results_saver, args=(child_conn, save_name)
    )
    saver.start()
    parent_conn.send(("save_config", (config,)))
    parent_conn.send(("reserve", ("score", n_pixel, False, None)))
    for name in species:
        parent_conn.send(
            ("reserve", (name, n_pixel, need_spectra, freq_data))
        )
    if len(species) > 1:
        parent_conn.send(
            ("reserve", ("total", n_pixel, need_spectra, freq_data))
        )

    loss_fn = config["cube"].get("loss_fn", "pm")
    if loss_fn == "chi2" and config["opt_single"]["method"] in ("trf", "dogbox", "lm"):
        loss_fn = "chi2_ls"
    with mp.Pool(config["n_process"]) as pool:
        if inf_model is None:
            if "x0" in config["cube"]:
                x0 = np.asarray(config["cube"]["x0"], dtype=np.floating)
            else:
                x0 = None
            fit_cube_optimize(
                fname_cube=fname_cube,
                slm_factory=slm_factory,
                postprocess=postprocess,
                loss_fn=loss_fn,
                species=species,
                x0=x0,
                conn=parent_conn,
                pool=pool
            )
        else:
            ai.predict_cube(
                inf_model=inf_model,
                postprocess=postprocess,
                loss_fn=loss_fn,
                fname_cube=fname_cube,
                species=species,
                batch_size=config["inference"]["batch_size"],
                num_workers=config["inference"]["num_workers"],
                conn=parent_conn,
                pool=pool,
                device=config["inference"]["device"]
            )

    parent_conn.send(("finish", None))
    saver.join()
    parent_conn.close()


def fit_cube_optimize(fname_cube: str,
                      slm_factory: SpectralLineModelFactory,
                      postprocess: Optimizer,
                      loss_fn: str,
                      species: str,
                      x0: Union[np.ndarray, None],
                      conn,
                      pool: mp.Pool):
    #
    def save_results(res):
        nonlocal idx_start
        res = res.get()
        if conn is None:
            results.extend(res)
        else:
            conn.send(("save", (idx_start, res)))
            idx_start += len(res)

    with h5py.File(fname_cube) as fp:
        n_pixel = fp["index"].shape[0]
    misc_data = load_misc_data(fname_cube)

    specie_list = []
    for id_, name in enumerate(species):
        specie_list.append({"id": id_, "root": name, "species": [name]})

    target = partial(
        fit_cube_worker,
        fname_cube, misc_data, slm_factory,
        postprocess, loss_fn, specie_list, x0,
    )

    batch_size = 32
    batch = [list(range(idx, min(idx + batch_size, n_pixel)))
             for idx in range(0, n_pixel, batch_size)]
    n_wait = 3
    results = []
    wait_list = []
    idx_start = 0
    with tqdm(total=len(batch), desc="Optimizing") as pbar:
        for inds in batch:
            wait_list.append(pool.map_async(target, inds))
            if len(wait_list) == n_wait:
                res = wait_list.pop(0)
                save_results(res)
                pbar.update()
        while len(wait_list) > 0:
            res = wait_list.pop(0)
            save_results(res)
            pbar.update()
    return results


def fit_cube_worker(fname_cube: str,
                    misc_data: list,
                    slm_factory: SpectralLineModelFactory,
                    postprocess: Callable,
                    loss_fn: str,
                    specie_list: list,
                    x0: Union[np.ndarray, None],
                    idx_pixel: int):
    """Basic function that performs fitting of one pixel in a cube."""
    obs_info = create_obs_info_from_cube(fname_cube, idx_pixel, misc_data)
    fitting_model = slm_factory.create_fitting_model(
        obs_info, specie_list, loss_fn
    )
    if x0 is None:
        return postprocess(fitting_model)
    return postprocess(fitting_model, x0)


def create_obs_info_from_cube(fname: str,
                              idx_pixel:int ,
                              misc_data: Optional[list]=None):
    T_obs_data = []
    T_bg_data = []
    with h5py.File(fname) as fp:
        for i_segment in range(len(fp["cube"])):
            grp = fp["cube"][str(i_segment)]
            T_obs_data.append(np.array(grp["T_obs"][idx_pixel]))
            T_bg_data.append(np.array(grp["T_bg"][idx_pixel]))
    if misc_data is None:
        freq_data, noise_data, beam_data = load_misc_data(fname)
    else:
        freq_data, noise_data, beam_data = misc_data

    obs_info = []
    for i_segment, freq in enumerate(freq_data):
        spec = np.vstack([freq, T_obs_data[i_segment]]).T
        spec = preprocess_spectrum(spec)
        obs_info.append({
            "spec": spec,
            "beam_info": beam_data[i_segment],
            "T_bg": T_bg_data[i_segment],
            "need_cmb": True,
            "noise": noise_data[i_segment],
        })
    return obs_info


def load_misc_data(fname):
    """
    Returns:
        freq_data (list):
        noise_data (list):
        beam_data (list):
    """
    with h5py.File(fname) as fp:
        freq_data = []
        noise_data = []
        beam_data = []
        for i_segment in range(len(fp["cube"])):
            grp = fp["cube"][str(i_segment)]
            freq_data.append(np.array(grp["freq"]))
            noise_data.append(np.array(grp["noise"]))
            beam_data.append(np.array(grp["beam"]))
    return freq_data, noise_data, beam_data


def load_cube_header(fname: str,
                     i_segment: int=0,
                     target: Literal["continuum", "line"]="continuum"):
    """Load the header of a cube file.

    Args:
        fname: Path to the input HDF5 cube file.
        i_segment: Segment index.
        target: Specify whether to read the header of the continuum or line
            file.

    Returns:
        dict: A dictionary containing all the header attributes.
    """
    with h5py.File(fname) as fp:
        grp = fp[f"cube/{i_segment}/header/{target}"]
        header = dict(grp.attrs)
    return header


def load_cube_units(fname: str):
    units = {}
    with h5py.File(fname) as fp:
        for key, val in fp.attrs.items():
            if key.startswith("unit/"):
                units[key[5:]] = val
    return units


def format_cube_results(results):
    res_dict = {}
    for res in results:
        name = res["specie"][0]["root"]
        if name not in res_dict:
            res_dict[name] = {
                "params": [],
                "t1_score": [],
                "t2_score": [],
                "t3_score": [],
                "t4_score": [],
                "s_tp_tot": [],
                "num_tp": [],
                "s_fp_tot": [],
                "num_fp": []
            }
            if "T_pred" in res:
                res_dict[name]["T_pred"] = {
                    f"{idx}": [] for idx in range(len(res["T_pred"]))
                }

        sub_dict = res_dict[name]
        for key in sub_dict:
            if key != "T_pred":
                sub_dict[key].append(res[key])
            elif "T_pred" in res:
                for idx, T_pred in enumerate(res["T_pred"]):
                    sub_dict["T_pred"][f"{idx}"].append(T_pred)

    # Convert to numpy arrays
    for sub_dict in res_dict.values():
        for key, val in sub_dict.items():
            if key == "params":
                sub_dict[key] = np.vstack(val)
            elif key == "T_pred":
                for idx, T_data in sub_dict[key].items():
                    sub_dict[key][idx] = np.vstack(T_data)
            else:
                sub_dict[key] = np.asarray(val)

    # Expand fitting parameters
    for sub_dict in res_dict.values():
        params = sub_dict.pop("params").T
        for key, arr in zip(PARAM_NAMES, params):
            sub_dict[key] = arr

    return res_dict


def _cube_results_saver(conn, save_name):
    with h5py.File(save_name, "w") as fp:
        for name, unit in zip(PARAM_NAMES, PARAM_UNITS):
            fp.attrs[f"unit/{name}"] = unit
        fp.attrs[f"unit/intensity"] = "K"
        fp.attrs[f"unit/frequency"] = "MHz"

        while True:
            task, data = conn.recv()
            if task == "finish":
                break
            elif task == "reserve":
                _save_empty_results(fp, *data)
            elif task == "save":
                _save_fitting_results(fp, *data)
            elif task == "save_config":
                config = data[0].copy()
                config["obs_info"] = None
                fp.attrs["config"] = json.dumps(config)
            else:
                raise ValueError(f"Unknown task: {task}")


def _save_empty_results(fp, group_name, n_pixel, need_spectra, freq_data):
    grp = fp.create_group(group_name)
    grp.attrs["type"] = "dict"

    if group_name == "score":
        grp.create_dataset("fun", shape=(n_pixel,), dtype="f4")
        grp.create_dataset("nfev", shape=(n_pixel,), dtype="i4")
        return grp

    if group_name != "total":
        for key in PARAM_NAMES:
            grp.create_dataset(key, shape=(n_pixel,), dtype="f4")

    #keys= (
    #    "t1_score", "t2_score", "t3_score", "t4_score",
    #    "s_tp_tot", "s_fp_tot"
    #)
    #for key in keys:
    #    grp.create_dataset(key, shape=(n_pixel,), dtype="f4")
    #keys = ("num_tp", "num_fp")
    #for key in keys:
    #    grp.create_dataset(key, shape=(n_pixel,), dtype="i4")

    if need_spectra:
        grp_sub = grp.create_group("T_pred")
        grp_sub.attrs["type"] = "list"
        for i_segment, freqs in enumerate(freq_data):
            grp_sub.create_dataset(
                f"{i_segment}", shape=(n_pixel, len(freqs)), dtype="f4"
            )
    return grp


def _save_fitting_results(fp, idx_start, results):
    idx = idx_start
    for res_dict in results:
        for name, res in res_dict.items():
            # Expand parameters
            if "params" in res:
                params = res.pop("params").T
                for key, arr in zip(PARAM_NAMES, params):
                    res[key] = arr

            grp = fp[name]
            for key in grp.keys():
                if key not in res:
                    continue

                val = res[key]
                if key == "T_pred":
                    for i_segment, T_pred in enumerate(val):
                        grp[f"T_pred/{i_segment}"][idx] = T_pred
                else:
                    grp[key][idx] = val
        idx += 1


class  _AddExtraProps:
    def __init__(self,
                 slm_factory: SpectralLineModelFactory,
                 postprocess: Callable,
                 need_spectra: bool):
        self._slm_factory = copy(slm_factory)
        self._slm_factory._sl_db = None # Prevent copying the database from multiprocessing
        self._postprocess = postprocess
        self._need_spectra = need_spectra

    def __call__(self, fitting_model, *args):
        ret_dict = {}
        res = self._postprocess(fitting_model, *args)
        specie_list = res["specie"]
        params = fitting_model.sl_model.param_mgr.derive_params(res["x"])
        # Compute individual spectra
        if self._need_spectra:
            T_single_dict = fitting_model.sl_model.compute_individual_spectra(res["x"])

        # Save total spectra if there are multiple species
        if self._need_spectra and len(specie_list) > 1:
            ret_dict["total"] = {"T_pred": res["T_pred"]}
        # Save parameters and individual spectra
        for item, params_sub in zip(specie_list, params, strict=True):
            res_sub = {}
            res_sub["params"] = params_sub
            if self._need_spectra:
                res_sub["T_pred"] = T_single_dict[item["root"]]
            ret_dict[item["root"]] = res_sub
        # Save fitting loss and number of function evaluations
        ret_dict["score"] = {"fun": res["fun"], "nfev": res["nfev"]}

        #peak_mgr = self._slm_factory.create_peak_mgr(fitting_model.obs_info)
        #scores_tp, scores_fp = peak_mgr.compute_score_all(
        #    res["T_pred"], use_f_dice=True
        #)
        #scores_tp = np.sort(scores_tp)[::-1]
        #n_max = 4
        #for idx in range(n_max):
        #    res[f"t{idx+1}_score"] \
        #        = scores_tp[idx] if len(scores_tp) > idx + 1 else 0.
        #res["s_tp_tot"] = np.sum(scores_tp)
        #res["num_tp"] = len(scores_tp)
        #res["s_fp_tot"] = np.sum(scores_fp)
        #res["num_fp"] = len(scores_fp)

        return ret_dict

    @property
    def n_draw(self):
        return self._postprocess.n_draw


@dataclass(frozen=True)
class CubeItem:
    line: str
    continuum: Optional[str] = None
    window: Optional[list] = None
    mask: Optional[list] = None
    header: Optional[dict] = None

    def __post_init__(self):
        if self.window is not None and self.mask is not None:
            raise ValueError("Can only specify either window or mask.")
        if self.window is not None:
            windows = check_overlap(self.window)
            if windows is None:
                raise ValueError("Windows have overlaps.")
        if self.mask is not None:
            masks = check_overlap(self.mask)
            if masks is None:
                raise ValueError("Masks have overlaps.")


@dataclass(frozen=True)
class CubePipeline:
    """Convert cube data into format that can be used by the model.

    For each segment, the pipline does the following steps:
        1. Set any inf and nan values in the spectrum to the continuum.
        2. Estimate the global RMS noise of the spectrum by randomly selecting
            several pixels and using sigma clipping.
        3. Set any outiler values (> 1e4*noise) to the continuum.
        4. Estimate the number of peaks in each spectrum.

    Secondly, the pipline removes bad fixels:
        1. Estimate the local RMS noise using the spectrum of one pixel.
        2. Count the number of peaks, and remove the pixel if the total number
            of peaks of all segments is below a threshold.
        3. Estimate the plateau fraction using by smoothing spectrum, and
            remove the pixel if the plateau fraction is above a threshold.

    Args:
        maxiters: Maximum number of iterations used in ``sigma_clipped_stats``.
        n_estimate: Number of pixels to estimate the RMS noise.
        atol_factor: Absolute tolerance factor to remove plateau.
        window_size: Window size to smooth the spectrum when removing plateau.
        f_pla_cut: Plateau fraction below which a pixel is removed.
        noise_factor_global: Global RMS noise multipler to identify peaks.
        noise_factor_local: Local RMS noise multipler to identify peaks.
        number_cut: Number of peak below which a pixel is removed.
    """
    maxiters: int = 1000
    n_estimate: int = 10000
    atol_factor: float = 1.e-4
    window_size: int = 31
    rel_height: float = 0.25
    noise_factor_global: float = 4.
    noise_factor_local: float = 6.
    number_cut: int = 3
    f_pla_cut: float = .75
    v_LSR: float = 0.

    def run(self,
            file_list: list[CubeItem],
            save_name: str,
            fname_mask=None,
            header_lookup: Optional[dict]=None):
        """Run the pipeline.

        Args:
            file_list: A list of dictionaries to specify the files of each
                spectral window, e.g.
                [
                    {
                        "continuum": PATH_TO_CONTIUUM_FILE_1,
                        "line": PATH_TO_LINE_FILE_1,
                    },
                    {
                        "continuum": PATH_TO_CONTIUUM_FILE_2,
                        "line": PATH_TO_LINE_FILE_2,
                    },
                ]
                The ``continumm`` key is optional. If not specified, the
                continuum will be set to zero.
            save_name: Saving name of the output HDF file.
            fname_mask: Path to a mask file. The file should have a 2D array.
                The masked pixels should be set to NaN.
            header_lookup: A dictionary to specify the aliases of some header
                attributes. Below is the default header lookup:
                {
                    "freq_start": "CRVAL3",
                    "dfreq": "CDELT3",
                    "n_freq": "NAXIS3",
                    "i_freq": "CRPIX3",
                    "BMAJ": "BMAJ",
                    "BMIN": "BMIN",
                }
                Change any attributes if necessary.
        """
        header_list = self.prepare_header_list(file_list, header_lookup)
        self.validate_freq_order(header_list)

        if fname_mask is None:
            cond_mask = None
        else:
            mask = np.squeeze(fits.open(fname_mask)[0].data)
            cond_mask = ~np.isnan(mask)

        for item in file_list:
            if item.continuum is None:
                continue
            data_continuum = np.squeeze(fits.open(item.continuum)[0].data) # (W, H)
            n_row, n_col = data_continuum.shape
            if cond_mask is None:
                cond_mask = np.full(data_continuum.shape, True)
            cond_mask &= ~np.isnan(data_continuum)
            cond_mask &= ~np.isinf(data_continuum)
            inds_mask = np.where(cond_mask)

        counts = None
        headers_line = []
        headers_continuum = []
        T_obs_data = []
        T_bg_data = []
        freq_data = []
        noise_data = []
        beam_data = []
        window_data = []
        mask_data = []
        shift_factor = 1. + const_factor_mu_sigma()[0]*self.v_LSR
        for item, header in zip(file_list, header_list):
            hdul = fits.open(item.line)[0]
            header_line = hdul.header
            # (1, C, W, H) > (C, W, H)
            data_line = np.squeeze(hdul.data)
            data_line = np.transpose(data_line, axes=(1, 2, 0)) # (W, H, C)
            if cond_mask is None:
                inds_mask = np.where(np.full(data_line.shape[:2], True))
                n_row, n_col = data_line.shape[:2]
            if counts is None:
                counts = np.zeros(len(inds_mask[0]), dtype="i8")
            data_line = data_line[inds_mask] # (N, C)

            if item.continuum is None:
                data_continuum = np.zeros(len(inds_mask[0]))
                header_cont = header_line
            else:
                hdul = fits.open(item.continuum)[0]
                header_cont = hdul.header
                data_continuum = np.squeeze(hdul.data) # (W, H)
                data_continuum = data_continuum[inds_mask]

            # Fix nan and inf values
            num_tot = np.prod(data_line.shape)
            num = 0
            for idx, spec in tqdm(
                enumerate(data_line),
                desc="Checking invalid values...",
                total=len(data_line)
            ):
                cond = np.isnan(spec) | np.isinf(spec)
                num += np.count_nonzero(cond)
                data_line[idx, cond] = data_continuum[idx]
            print("Fraction of nan and inf values: {:.1f}% ({}/{})".format(
                num/num_tot*100., num, num_tot))

            # Estimate RMS noise
            print("Estimating noise...")
            noise = self.estimate_noise(data_line)
            freqs = self.load_freqs(header)
            freq_m = .5*(freqs[0] + freqs[-1])
            bmaj = header["BMAJ"]
            bmin = header["BMIN"]

            #
            if freqs[1] < freqs[0]:
                freqs = freqs[::-1]
                data_line = data_line[:, ::-1]

            # Fix outliers
            factor_out = 1e4
            num = 0
            for idx, spec in tqdm(
                enumerate(data_line),
                desc="Checking invalid values...",
                total=len(data_line)
            ):
                cond = spec > factor_out*noise
                num += np.count_nonzero(cond)
                data_line[idx, cond] = data_continuum[idx]
            print("Fraction of outliers: {:.1f}% ({}/{})".format(
                num/num_tot*100., num, num_tot))

            # Count number of lines
            prominence = self.noise_factor_global*noise
            pbar = tqdm(
                enumerate(data_line),
                desc="Counting peaks...",
                total=len(data_line)
            )
            for idx, spec in pbar:
                peaks = signal.find_peaks(
                    np.maximum(spec, 0.), # Neglect absoprtion
                    height=prominence,
                    prominence=prominence,
                )[0]
                counts[idx] += len(peaks)

            # Convert unit
            T_bg = to_kelvin(data_continuum, freq_m, bmaj, bmin)
            T_obs = to_kelvin(data_line, freqs, bmaj, bmin)
            noise = to_kelvin(noise, freq_m, bmaj, bmin)
            beam_info = np.array([bmaj, bmin])

            # Prepare data to be saved
            T_obs_data.append(T_obs)
            T_bg_data.append(T_bg)
            freq_data.append(freqs)
            noise_data.append(noise)
            beam_data.append(beam_info)

            n_sub = 1
            if item.window is not None:
                windows = [[shift_factor*x[0], shift_factor*x[1]]
                           for x in item.window]
                window_data.append(windows)
                n_sub = len(windows)
            else:
                window_data.append(None)
            if item.mask is not None:
                masks = [[shift_factor*x[0], shift_factor*x[1]]
                         for x in item.mask]
                mask_data.append(masks)
                n_sub = len(masks) + 1
            else:
                mask_data.append(None)

            headers_line.extend([header_line]*n_sub)
            headers_continuum.extend([header_cont]*n_sub)

        #
        T_obs_data, \
        T_bg_data, \
        freq_data, \
        noise_data, \
        beam_data = self.extract_sub_window(
            T_obs_data, T_bg_data, freq_data,
            noise_data, beam_data, window_data, mask_data,
        )

        #
        cond = self.filter_pixel(T_obs_data, noise_data, counts)
        if np.count_nonzero(cond) == 0:
            print("No acceptable pixels in {}.".format(save_name))
            return
        T_obs_data = [T_obs_cube[cond] for T_obs_cube in T_obs_data]
        T_bg_data = [T_bg_cube[cond] for T_bg_cube in T_bg_data]
        counts = counts[cond]
        inds_mask = np.vstack(inds_mask).T
        inds_mask = inds_mask[cond]

        print("Saving data...")
        with h5py.File(save_name, "w") as fp:
            fp.attrs["n_row"] = n_row
            fp.attrs["n_col"] = n_col
            fp.attrs["max_count"] = max(counts)
            fp.attrs[f"unit/intensity"] = "K"
            fp.attrs[f"unit/frequency"] = "MHz"
            for key, val in asdict(self).items():
                fp.attrs[key] = val
            fp.create_dataset("count", data=counts, dtype="i4")
            fp.create_dataset("index", data=inds_mask, dtype="i4")
            grp_cube = fp.create_group("cube")
            self.save_file(
                grp_cube, headers_line, headers_continuum,
                T_obs_data, T_bg_data, freq_data, noise_data, beam_data
            )

    def prepare_header_list(self,
                            file_list: list[CubeItem],
                            header_lookup: dict):
        header_lookup_ = {
            "freq_start": "CRVAL3",
            "dfreq": "CDELT3",
            "n_freq": "NAXIS3",
            "i_freq": "CRPIX3",
            "BMAJ": "BMAJ",
            "BMIN": "BMIN",
        }
        if header_lookup is not None:
            header_lookup_.update(header_lookup)

        header_list = []
        for item in file_list:
            header_aux = item.header if item.header is not None else {}
            if item.continuum is None:
                header_cont = {}
            else:
                header_cont = dict(fits.open(item.continuum)[0].header)
            header_line = dict(fits.open(item.line)[0].header)
            header = {}
            for key, alias in header_lookup_.items():
                if key in header_aux:
                    header[key] = header_aux[key]
                elif alias in header_line:
                    header[key] = header_line[alias]
                elif alias in header_cont:
                    header[key] = header_cont[alias]
                elif alias == "i_freq":
                    header[key] = 1
                else:
                    raise ValueError("Cannot find {} in header.".format(key))

            header_list.append(header)
        return header_list

    def validate_freq_order(self, header_list):
        freqs = []
        for header in header_list:
            freqs.append(header["freq_start"])
        msg = "Frequencies of the segments must be in ascending order."
        assert np.all(np.diff(freqs) > 0), msg

    def load_freqs(self, header):
        """Load frequency data from a FITS header.

        Args:
            header (object): FITS header.

        Returns:
            np.ndarray: Frequency data (MHz).
        """
        dfreq = header["dfreq"]
        freq_start = header["freq_start"] + (1 - header["i_freq"])*dfreq
        num = header["n_freq"]
        freq_end = freq_start + (num - 1)*dfreq
        freqs = np.linspace(freq_start, freq_end, num)
        freqs *= 1e-6 # Hz to MHz
        freqs *= (1. + const_factor_mu_sigma()[0]*self.v_LSR)
        return freqs

    def estimate_noise(self, data):
        if len(data) > self.n_estimate:
            inds = np.random.choice(len(data), self.n_estimate, replace=False)
            data_sub = data[inds]
        else:
            data_sub = data
        return sigma_clipped_stats(
            np.ravel(data_sub), maxiters=self.maxiters, stdfunc="mad_std"
        )[-1]

    def filter_pixel(self, T_obs_data, noise_data, counts):
        number_cut = self.number_cut*len(T_obs_data)
        cond = counts >= number_cut
        inds = np.where(cond)[0]
        counts_sub = np.zeros(len(cond), dtype="i4")
        for T_obs_cube, noise in tqdm(
            zip(T_obs_data, noise_data),
            desc="Filtering pixels...",
            total=len(T_obs_data)
        ):
            f_pla = np.ones(len(cond))
            for idx, T_obs in zip(inds, T_obs_cube[inds]):
                T_obs, f_pla[idx] = self.remove_plateau(
                    T_obs, self.atol_factor*noise, self.window_size
                )
                _, median, std = sigma_clipped_stats(
                    T_obs, maxiters=self.maxiters, stdfunc="mad_std"
                )
                peaks, _ = signal.find_peaks(
                    T_obs - median,
                    height=self.noise_factor_local*std,
                    prominence=self.noise_factor_local*std,
                    rel_height=self.rel_height
                )
                counts_sub[idx] += len(peaks)
            cond &= f_pla >= self.f_pla_cut
        cond &= counts_sub >= number_cut
        return cond

    def remove_plateau(self, target, atol, window_size):
        window = np.ones(window_size)/window_size
        smoothed = np.convolve(target, window, mode="same")
        cond = np.abs(target - smoothed) > atol
        target_ret = target[cond]
        frac = np.count_nonzero(cond)/len(target)
        return target_ret, frac

    def extract_sub_window(self, T_obs_data, T_bg_data, freq_data,
                           noise_data, beam_data, window_data, mask_data):
        window_data_ = []
        for windows, masks in zip(window_data, mask_data):
            if windows is not None:
                window_data_.append(windows)
            elif masks is not None:
                windows = []

                left = masks[0][0]
                windows.append((-np.inf, left))

                for idx in range(1, len(masks)):
                    right_prev = masks[idx - 1][1]
                    left = masks[idx][0]
                    windows.append((right_prev, left))

                windows.append((masks[-1][1], np.inf))
                window_data_.append(windows)
            else:
                window_data_.append(None)

        T_obs_data_ret = []
        T_bg_data_ret = []
        freq_data_ret = []
        noise_data_ret = []
        beam_data_ret = []
        for T_obs, T_bg, freqs, noise, beam, windows in zip(
            T_obs_data, T_bg_data, freq_data, noise_data, beam_data, window_data_
        ):
            if windows is None:
                T_obs_data_ret.append(T_obs)
                T_bg_data_ret.append(T_bg)
                freq_data_ret.append(freqs)
                noise_data_ret.append(noise)
                beam_data_ret.append(beam)
            else:
                for left, right in windows:
                    cond = (freqs >= left) & (freqs <= right)
                    T_obs_data_ret.append(T_obs[..., cond])
                    T_bg_data_ret.append(T_bg)
                    freq_data_ret.append(freqs[cond])
                    noise_data_ret.append(noise)
                    beam_data_ret.append(beam)

        return T_obs_data_ret, T_bg_data_ret, freq_data_ret, noise_data_ret, beam_data_ret

    def save_file(self, grp, headers_line, headers_continuum,
                  T_obs_data, T_bg_data, freq_data, noise_data, beam_data):
        i_seg_save = 0
        for header_l, header_c, T_obs, T_bg, freqs, noise, beam in zip(
            headers_line, headers_continuum,
            T_obs_data, T_bg_data, freq_data, noise_data, beam_data,
            strict=True,
        ):
            grp_sub = grp.create_group(str(i_seg_save))
            grp_header_l = grp_sub.create_group("header/line")
            for key, val in header_l.items():
                grp_header_l.attrs[key] = val
            grp_header_c = grp_sub.create_group("header/continuum")
            for key, val in header_c.items():
                grp_header_c.attrs[key] = val
            grp_sub.create_dataset("T_obs", data=T_obs)
            grp_sub.create_dataset("T_bg", data=T_bg, dtype="f4")
            grp_sub.create_dataset("freq", data=freqs, dtype="f4")
            grp_sub.create_dataset("noise", data=noise, dtype="f4")
            grp_sub.create_dataset("beam", data=beam, dtype="f4")
            i_seg_save += 1


def to_kelvin(J_obs, freqs, bmaj, bmin):
    """Convert Jansky/beam to K.

    Args:
        T_obs (np.ndarray): Jansky.
        freqs (np.ndarray): MHz.
        bmaj (float): Degree.
        bmin (float): Degree.
    """
    factor = constants.c**2/(2*constants.k_B)/(np.pi/(4*np.log(2)))*units.steradian
    factor = factor.to(units.Kelvin*units.MHz**2/units.Jy*units.degree**2).value
    return factor/(freqs*freqs*bmaj*bmin)*J_obs


class HDFCubeManager:
    def __init__(self, fname: str):
        self._fname = fname
        self._freq_data = load_misc_data(self._fname)[0]
        with h5py.File(self._fname, "r") as fp:
            self._indices = np.array(fp["index"])
            self._shape = fp.attrs["n_row"], fp.attrs["n_col"]

    def load_count_map(self):
        """Load the map which shows the number of peaks in each pixel."""
        with h5py.File(self._fname, "r") as fp:
            count = np.array(fp["count"], dtype="f4")
        return to_dense_matrix(count, self._indices, self._shape)

    def load_pred_data(self, fname: str, target: str):
        with h5py.File(fname, "r") as fp:
            data = np.array(fp[target])
        return to_dense_matrix(data, self._indices, self._shape)

    def obs_data_to_fits(self,
                         save_dir: str,
                         add_T_bg: bool=False,
                         overwrite=False):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        units = load_cube_units(self._fname)
        for i_segment in range(len(self._freq_data)):
            # Save T_bg
            header_conti = self._derive_header_scalar(i_segment)
            with h5py.File(self._fname) as fp:
                T_bg = np.array(fp[f"cube/{i_segment}/T_bg"])
            T_bg = to_dense_matrix(T_bg, self._indices, self._shape)
            T_bg = T_bg.astype("f4")
            header_conti["BUNIT"] = units["intensity"]
            hdu = fits.PrimaryHDU(T_bg, header=header_conti)
            save_name = save_dir/f"{i_segment}_obs_continuum.fits"
            hdu.writeto(save_name, overwrite=overwrite)

            # Save T_obs
            with h5py.File(self._fname) as fp:
                T_obs = np.array(fp[f"cube/{i_segment}/T_obs"])
            T_obs = to_dense_matrix(T_obs, self._indices, self._shape)
            T_obs = np.transpose(T_obs, axes=(2, 0, 1))
            if add_T_bg:
                T_obs += T_bg
            T_obs = T_obs.astype("f4")
            header_line = self._derive_header_line(i_segment)
            header_line["BUNIT"] = units["intensity"]
            header_line["CUNIT3"] = units["frequency"]
            hdu = fits.PrimaryHDU(T_obs, header=header_line)
            save_name = save_dir/f"{i_segment}_obs_line.fits"
            hdu.writeto(save_name, overwrite=overwrite)

    def pred_data_to_fits(self,
                          fname: str,
                          save_dir: str,
                          add_T_bg: bool=False,
                          overwrite=False):
        units = load_cube_units(fname)
        with h5py.File(fname) as fp:
            for name, grp in fp.items():
                self._save_pred_data(
                    name=name,
                    grp=grp,
                    units=units,
                    save_dir=save_dir,
                    add_T_bg=add_T_bg,
                    overwrite=overwrite
                )

    def _save_pred_data(self, name, grp, units, save_dir, add_T_bg, overwrite):
        save_dir = Path(save_dir)/name
        save_dir.mkdir(parents=True, exist_ok=True)

        header = self._derive_header_scalar(i_segment=0)
        for key, data in grp.items():
            if key == "T_pred":
                continue

            data = np.array(data)
            data = to_dense_matrix(data, self._indices, self._shape)
            if key in units:
                header["BUNIT"] = units[key]
            else:
                header["BUNIT"] = "none"
            hdu = fits.PrimaryHDU(data, header=header)
            save_name = save_dir/f"{key}.fits"
            hdu.writeto(save_name, overwrite=overwrite)

        if "T_pred" not in grp:
            return

        for i_segment in range(len(self._freq_data)):
            T_pred = np.array(grp[f"T_pred/{i_segment}"])
            T_pred = to_dense_matrix(T_pred, self._indices, self._shape)
            T_pred = np.transpose(T_pred, axes=(2, 0, 1))
            if add_T_bg:
                with h5py.File(self._fname) as fp:
                    T_bg = np.array(fp[f"cube/{i_segment}/T_bg"])
                T_bg = to_dense_matrix(T_bg, self._indices, self._shape)
                T_pred += T_bg
            header_line = self._derive_header_line(i_segment)
            header_line["BUNIT"] = units["intensity"]
            header_line["CUNIT3"] = units["frequency"]
            hdu = fits.PrimaryHDU(T_pred, header=header_line)
            save_name = save_dir/f"{i_segment}_line.fits"
            hdu.writeto(save_name, overwrite=overwrite)

    def _derive_header_line(self, i_segment):
        freqs = self._freq_data[i_segment]
        header = load_cube_header(self._fname, i_segment, "line")
        header = fits.Header(header)
        header["CRVAL3"] = freqs[0]
        header["CDELT3"] = (freqs[-1] - freqs[0])/(len(freqs) - 1)
        header["ORIGIN"] = f"Spectuner {__version__}"
        return header

    def _derive_header_scalar(self, i_segment):
        header = load_cube_header(self._fname, i_segment, "continuum")
        header = fits.Header(header)
        header["ORIGIN"] = f"Spectuner {__version__}"
        return header


def to_dense_matrix(arr: np.ndarray,
                    indices: np.ndarray,
                    shape: Optional[tuple]=None):
    """Covnert input array to dense matrix using indices.

    Args:
        arr:  Input array, whose shape can be (N,) or (N, D). If the shape is
            (N, D), the output matrix will have shape (W, H, D).
        indices: Indices (N, 2).
        shape: Shape of the output matrix. If ``None``, the shape will be
            inferred from the maximum indices.

    Returns:
        Dense matrix (W, H) or (W, H, D).
    """
    if shape is None:
        shape = np.max(indices, axis=0) + 1
    if len(arr.shape) > 1:
        shape = (*shape, *arr.shape[1:])

    if np.issubdtype(arr.dtype, np.floating):
        mat = np.full(shape, np.nan, dtype=arr.dtype)
    else:
        mat = np.full(shape, 0, dtype=arr.dtype)
    mat[indices[:, 0], indices[:, 1]] = arr
    return mat


def ra_dec_plots(header, naxis=2, **kwargs):
    wcs = WCS(header, naxis=naxis)
    fig, axes = plt.subplots(**kwargs, subplot_kw={'projection': wcs})
    for ax in np.ravel(axes):
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_axislabel('RA (J2000)')
        lat.set_axislabel('DEC (J2000)')
        lon.set_major_formatter('hh:mm:ss.s')
    return fig, axes