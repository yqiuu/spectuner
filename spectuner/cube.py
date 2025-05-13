from __future__ import annotations
import multiprocessing as mp
from typing import Optional, Callable, Literal
from itertools import product
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .sl_model import (
    create_spectral_line_db,
    query_species,
    const_factor_mu_sigma,
    SQLSpectralLineDB
)
from .slm_factory import SpectralLineModelFactory
from .optimize import create_optimizer, Optimizer
from .ai import collate_fn_padding, InferenceModel
from .preprocess import preprocess_spectrum
from .utils import hdf_save_dict


PARAM_NAMES = ("theta", "T_ex", "N_tot", "delta_v", "v_offset")
PARAM_UNITS = ("arcsec", "K", "cm-2", "km/s", "km/s")


def fit_cube(config: dict,
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
            - Set opt_single/ for optimization configuration.
            - Set inference/ for AI model configuration.

        fname_cube: Path to the cube data.
        save_name: Path to the output file.
        sl_db: Input spectroscopic database. If ``None``, load the one specified
            in sl_model/fname_db.
    """
    if sl_db is None:
        sl_db = create_spectral_line_db(config["sl_model"]["fname_db"])

    assert not config["species"]["combine_iso"] \
        and not config["species"]["combine_state"], \
        "Set combine_iso and combine_state to False for this function"

    freq_data = load_misc_data(fname_cube)[0]
    groups, _ = query_species(
        sl_db=sl_db,
        freq_data=freq_data,
        v_range=config["bound_info"]["v_LSR"],
        **config["species"],
    )
    species = [grp[0] for grp in groups]

    max_species = config["species"].get("max_species", 10)
    assert len(species) < max_species, \
        f"Number of species must be less than {max_species}"

    if config["inference"]["ckpt"] is None:
        inf_model = None
        slm_factory = SpectralLineModelFactory(config, sl_db=sl_db)
    else:
        inf_model = InferenceModel.from_config(config, sl_db=sl_db)
        slm_factory = inf_model.slm_factory
    opt = create_optimizer(config["opt_single"])
    postprocess = _AddExtraProps(
        slm_factory, opt, need_spectra=config["cube"]["need_spectra"]
    )
    n_process = config["opt_single"]["n_process"]
    with mp.Pool(processes=n_process) as pool:
        for specie_name in species:
            if inf_model is None:
                results = fit_cube_optimize(
                    fname_cube=fname_cube,
                    slm_factory=slm_factory,
                    postprocess=postprocess,
                    species=[specie_name],
                    pool=pool
                )
            else:
                results = predict_cube(
                    inf_model=inf_model,
                    postprocess=postprocess,
                    fname_cube=fname_cube,
                    species=[specie_name],
                    batch_size=config["inference"]["batch_size"],
                    num_workers=config["inference"]["num_workers"],
                    pool=pool,
                    device=config["inference"]["device"]
                )
            res_dict = format_cube_results(results)
    with h5py.File(save_name, "w") as fp:
        for name, unit in zip(PARAM_NAMES, PARAM_UNITS):
            fp.attrs[f"unit/{name}"] = unit
        fp.attrs[f"unit/intensity"] = "K"
        fp.attrs[f"unit/frequency"] = "MHz"
        hdf_save_dict(fp, res_dict)


def fit_cube_optimize(fname_cube: str,
                      slm_factory: SpectralLineModelFactory,
                      postprocess: Optimizer,
                      species: str,
                      pool: mp.Pool):
    with h5py.File(fname_cube) as fp:
        n_pixel = fp["index"].shape[0]
    misc_data = load_misc_data(fname_cube)
    args = fname_cube, misc_data, slm_factory, postprocess
    results = []
    with tqdm(total=n_pixel*len(species), desc="Optimizing") as pbar:
        callback = lambda _: pbar.update()
        for specie_name, idx_pixel in product(species, range(n_pixel)):
            specie_list = [
                {"id": 0, "root": specie_name, "species": [specie_name]}
            ]
            results.append(pool.apply_async(
                fit_cube_worker,
                args=(*args, specie_list, idx_pixel),
                callback=callback
            ))
        results = [res.get() for res in results]
    return results


def fit_cube_worker(fname_cube: str,
                    misc_data: list,
                    slm_factory: SpectralLineModelFactory,
                    postprocess: Callable,
                    specie_list: list,
                    idx_pixel: int):
    """Basic function that performs fitting of one pixel in a cube."""
    obs_info = create_obs_info_from_cube(fname_cube, idx_pixel, misc_data)
    fitting_model = slm_factory.create_fitting_model(obs_info, specie_list)
    return postprocess(fitting_model)


def predict_cube(inf_model: InferenceModel,
                 fname_cube: str,
                 species: str,
                 batch_size: int,
                 postprocess: Callable,
                 num_workers: int=2,
                 pool=None,
                 device=None):
    # Create data loader
    dataset = CubeDataset(
        fname=fname_cube,
        species=species,
        inf_model=inf_model,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_padding,
        shuffle=False,
        num_workers=num_workers,
    )
    return inf_model.call_multi(data_loader, postprocess, pool, device)


def create_obs_info_from_cube(fname: str, idx_pixel:int , misc_data: list):
    T_obs_data = []
    T_bg_data = []
    with h5py.File(fname) as fp:
        for i_segment in range(len(fp["cube"])):
            grp = fp["cube"][str(i_segment)]
            T_obs_data.append(np.array(grp["T_obs"][idx_pixel]))
            T_bg_data.append(np.array(grp["T_bg"][idx_pixel]))
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


class CubeDataset(Dataset):
    def __init__(self, fname: str, species: str, inf_model: InferenceModel):
        self._fname = fname
        self._species = species
        self._embedding_model = inf_model.embedding_model
        self._slm_factory = inf_model.slm_factory
        self._misc_data = load_misc_data(fname)
        self._n_pixel = h5py.File(fname)["index"].shape[0]

    def __len__(self):
        return self._n_pixel*len(self._species)

    def __getitem__(self, idx):
        idx_specie, idx_pixel = divmod(idx, self.n_pixel)
        obs_info = create_obs_info_from_cube(
            self._fname, idx_pixel, self._misc_data
        )
        embed_obs, embed_sl, sl_dict, specie_list \
            = self._embedding_model(obs_info, self._species[idx_specie])
        fitting_model = self._slm_factory.create_fitting_model(
            obs_info, specie_list, [sl_dict]
        )
        return embed_obs, embed_sl, fitting_model

    @property
    def n_pixel(self):
        return self._n_pixel


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
        res = self._postprocess(fitting_model, *args)
        params = fitting_model.sl_model.param_mgr.derive_params(res["x"])[0]
        res["params"] = params
        peak_mgr = self._slm_factory.create_peak_mgr(fitting_model.obs_info)
        scores_tp, scores_fp = peak_mgr.compute_score_all(
            res["T_pred"], use_f_dice=True
        )
        scores_tp = np.sort(scores_tp)[::-1]
        n_max = 4
        for idx in range(n_max):
            res[f"t{idx+1}_score"] \
                = scores_tp[idx] if len(scores_tp) > idx + 1 else 0.
        res["s_tp_tot"] = np.sum(scores_tp)
        res["num_tp"] = len(scores_tp)
        res["s_fp_tot"] = np.sum(scores_fp)
        res["num_fp"] = len(scores_fp)
        if not self._need_spectra:
            del res["T_pred"]
        return res

    @property
    def n_draw(self):
        return self._postprocess.n_draw


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
            file_list: list,
            header_lookup: dict,
            save_name: str,
            fname_mask=None):
        header_list = self.prepare_header_list(file_list, header_lookup)
        self.validate_freq_order(header_list)

        counts = None
        if fname_mask is None:
            cond_mask = None
        else:
            mask = np.squeeze(fits.open(fname_mask)[0].data)
            cond_mask = ~np.isnan(mask)

        for item in file_list:
            data_continuum = np.squeeze(fits.open(item["continuum"])[0].data) # (W, H)
            n_row, n_col = data_continuum.shape
            if cond_mask is None:
                cond_mask = np.full(data_continuum.shape, True)
            cond_mask &= ~np.isnan(data_continuum)
            cond_mask &= ~np.isinf(data_continuum)
        inds_mask = np.where(cond_mask)

        headers_line = []
        headers_continuum = []
        T_obs_data = []
        T_bg_data = []
        freq_data = []
        noise_data = []
        beam_data = []
        mask_data = []
        for item, header in zip(file_list, header_list):
            hdul = fits.open(item["line"])[0]
            headers_line.append(hdul.header)
            # (1, C, W, H) > (C, W, H)
            data_line = np.squeeze(hdul.data)
            data_line = np.transpose(data_line, axes=(1, 2, 0)) # (W, H, C)
            data_line = data_line[inds_mask] # (N, C)

            hdul = fits.open(item["continuum"])[0]
            headers_continuum.append(hdul.header)
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
            counts = np.zeros(len(data_continuum), dtype="i8")
            prominence = self.noise_factor_global*noise
            pbar = tqdm(
                enumerate(data_line),
                desc="Counting peaks...",
                total=len(data_line)
            )
            for idx, spec in pbar:
                peaks = signal.find_peaks(
                    np.maximum(spec, 0.), # Neglect absoprtion
                    prominence=prominence,
                    rel_height=self.rel_height
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
            mask_data.append(item.get("mask", None))

        #
        T_obs_data, \
        T_bg_data, \
        freq_data, \
        noise_data, \
        beam_data = self.split_by_mask(
            T_obs_data, T_bg_data, freq_data, noise_data, beam_data, mask_data
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
            for key, val in asdict(self).items():
                fp.attrs[key] = val
            fp.create_dataset("count", data=counts, dtype="i4")
            fp.create_dataset("index", data=inds_mask, dtype="i4")
            grp_cube = fp.create_group("cube")
            self.save_file(
                grp_cube, headers_line, headers_continuum,
                T_obs_data, T_bg_data, freq_data, noise_data, beam_data
            )

    def prepare_header_list(self, file_list, header_lookup):
        header_list = []
        names = ["freq_start", "dfreq", "n_freq", "BMAJ", "BMIN"]
        for item in file_list:
            header_aux = item.get("header", {})
            header_cont = dict(fits.open(item["continuum"])[0].header)
            header_line = dict(fits.open(item["line"])[0].header)

            header = {}
            for key in names:
                if key in header_lookup:
                    alias = header_lookup[key]
                else:
                    alias = key

                if key in header_aux:
                    header[key] = header_aux[key]
                elif alias in header_line:
                    header[key] = header_line[alias]
                elif alias in header_cont:
                    header[key] = header_cont[alias]
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
        """Load frequency data from a FITS file.

        Args:
            data (object): FITS file.

        Returns:
            np.ndarray: Frequency data (MHz).
        """
        freq_start = header["freq_start"]
        dfreq = header["dfreq"]
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
        cond = counts >= self.number_cut
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
        cond &= counts_sub >= self.number_cut
        return cond

    def remove_plateau(self, target, atol, window_size):
        window = np.ones(window_size)/window_size
        smoothed = np.convolve(target, window, mode="same")
        cond = np.abs(target - smoothed) > atol
        target_ret = target[cond]
        frac = np.count_nonzero(cond)/len(target)
        return target_ret, frac

    def split_by_mask(self, T_obs_data, T_bg_data, freq_data, noise_data, beam_data, mask_data):
        T_obs_data_ret = []
        T_bg_data_ret = []
        freq_data_ret = []
        noise_data_ret = []
        beam_data_ret = []
        for T_obs, T_bg, freqs, noise, beam, masks in zip(
            T_obs_data, T_bg_data, freq_data, noise_data, beam_data, mask_data
        ):
            if masks is None:
                T_obs_data_ret.append(T_obs)
                T_bg_data_ret.append(T_bg)
                freq_data_ret.append(freqs)
                noise_data_ret.append(noise)
                beam_data_ret.append(beam)
            else:
                masks.sort(key=lambda x: x[0])
                cond_r = np.full(len(freqs), True)
                upper_prev = -np.inf
                while len(masks) > 0:
                    lower, upper = masks.pop(0)
                    if upper_prev < lower:
                        cond_l = (freqs >= upper_prev) & (freqs <= lower)
                        T_obs_data_ret.append(T_obs[..., cond_l])
                        T_bg_data_ret.append(T_bg)
                        freq_data_ret.append(freqs[cond_l])
                        noise_data_ret.append(noise)
                        beam_data_ret.append(beam)
                    upper_prev = upper

                if upper_prev < freqs[-1]:
                    cond_r = freqs > upper_prev
                    T_obs_data_ret.append(T_obs[..., cond_r])
                    T_bg_data_ret.append(T_bg)
                    freq_data_ret.append(freqs[cond_r])
                    noise_data_ret.append(noise)
                    beam_data_ret.append(beam)

        return T_obs_data_ret, T_bg_data_ret, freq_data_ret, noise_data_ret, beam_data_ret

    def save_file(self, grp, headers_line, headers_continuum,
                  T_obs_data, T_bg_data, freq_data, noise_data, beam_data):
        i_seg_save = 0
        for header_l, header_c, T_obs, T_bg, freqs, noise, beam in zip(
            headers_line, headers_continuum,
            T_obs_data, T_bg_data, freq_data, noise_data, beam_data
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


class CubeConverter:
    """Convert HDF cube to FITS files.

    Args:
        fname: Path to the HDF file that saves the observed data.
    """
    def __init__(self, fname: str, save_dir: str="./"):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self._fname = fname
        self._save_dir = save_dir
        self._freq_data = load_misc_data(self._fname)[0]
        with h5py.File(self._fname, "r") as fp:
            self._indices = np.array(fp["index"])
            self._shape = fp.attrs["n_row"], fp.attrs["n_col"]

    def save_obs_data(self, overwrite=False):
        for i_segment in range(len(self._freq_data)):
            # Save T_obs
            with h5py.File(self._fname) as fp:
                T_obs = np.array(fp[f"cube/{i_segment}/T_obs"])
            T_obs = to_dense_matrix(T_obs, self._indices, self._shape)
            T_obs = np.transpose(T_obs, axes=(2, 0, 1))
            T_obs = T_obs.astype("f4")
            header_line = self._derive_header_line(i_segment)
            hdu = fits.PrimaryHDU(T_obs, header=header_line)
            save_name = self._save_dir/f"{i_segment}_obs_line.fits"
            hdu.writeto(save_name, overwrite=overwrite)

            # Save T_bg
            header_conti = self._derive_header_scalar(i_segment)
            with h5py.File(self._fname) as fp:
                T_bg = np.array(fp[f"cube/{i_segment}/T_bg"])
                header_conti["BUNIT"] = "K"
            hdu = fits.PrimaryHDU(T_bg, header=header_conti)
            save_name = self._save_dir/f"{i_segment}_obs_continuum.fits"
            hdu.writeto(save_name, overwrite=overwrite)

    def save_pred_data(self, fname, overwrite=False):
        with h5py.File(fname) as fp:
            unit_dict = {}
            for key, val in fp.attrs.items():
                if key.startswith("unit/"):
                    unit_dict[key[5:]] = val
            for name, grp in fp.items():
                self._save_pred_data_sub(name, grp, unit_dict, overwrite)

    def _save_pred_data_sub(self, name, grp, unit_dict, overwrite):
        save_dir = self._save_dir/name
        save_dir.mkdir(parents=True, exist_ok=True)

        header = self._derive_header_scalar(i_segment=0)
        for key, data in grp.items():
            if key == "T_pred":
                continue

            data = np.array(data)
            data = to_dense_matrix(data, self._indices, self._shape)
            if key in unit_dict:
                header["BUNIT"] = unit_dict[key]
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

            header_line = self._derive_header_line(i_segment)
            hdu = fits.PrimaryHDU(T_pred, header=header_line)
            save_name = save_dir/f"{i_segment}_line.fits"
            hdu.writeto(save_name, overwrite=overwrite)

    def _derive_header_line(self, i_segment):
        freqs = self._freq_data[i_segment]
        header = load_cube_header(self._fname, i_segment, "line")
        header = fits.Header(header)
        header["BUNIT"] = "K"
        header["CUNIT3"] = "MHz"
        header["CRVAL3"] = freqs[0]
        header["CDELT3"] = freqs[1] - freqs[0]
        return header

    def _derive_header_scalar(self, i_segment):
        header = load_cube_header(self._fname, i_segment, "continuum")
        header = fits.Header(header)
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