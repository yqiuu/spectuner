from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

import h5py
import numpy as np
from scipy import signal
from astropy import units, constants
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from tqdm import tqdm

from .sl_model import const_factor_mu_sigma


def convert_cube_data(cube_helper: CubeHelper,
                      file_list: list,
                      header_lookup: dict,
                      save_name: str,
                      noise_factor,
                      rel_height: float= 0.5,
                      fname_mask=None,
                      v_LSR: float=0.,
                      kwargs_filter=None):
    if kwargs_filter is None:
        kwargs_filter = {}

    header_list = cube_helper.prepare_header_list(file_list, header_lookup)
    cube_helper.validate_freq_order(header_list)

    counts = None
    if fname_mask is None:
        cond_mask = None
    else:
        mask = np.squeeze(fits.open(fname_mask)[0].data)
        cond_mask = ~np.isnan(mask)

    for item in file_list:
        data_contiuum = np.squeeze(fits.open(item["contiuum"])[0].data) # (W, H)
        n_row, n_col = data_contiuum.shape
        if cond_mask is None:
            cond_mask = np.full(data_contiuum.shape, True)
        cond_mask &= ~np.isnan(data_contiuum)
        cond_mask &= ~np.isinf(data_contiuum)
    inds_mask = np.where(cond_mask)

    T_obs_data = []
    T_bg_data = []
    freq_data = []
    noise_data = []
    beam_data = []
    mask_data = []
    for item, header in zip(file_list, header_list):
        hdul = fits.open(item["line"])[0]
        # (1, C, W, H) > (C, W, H)
        data_line = np.squeeze(hdul.data)
        data_line = np.transpose(data_line, axes=(1, 2, 0))
        data_line = data_line[inds_mask]

        hdul = fits.open(item["contiuum"])[0]
        data_contiuum = np.squeeze(hdul.data) # (W, H)
        data_contiuum = data_contiuum[inds_mask]

        # Fix nan and inf values
        n_pixel = len(inds_mask[0])
        num = 0
        for idx, spec in tqdm(
            enumerate(data_line),
            desc="Checking invalid values...",
            total=len(data_line)
        ):
            cond = np.isnan(spec) | np.isinf(spec)
            num += np.count_nonzero(cond)
            data_line[idx, cond] = data_contiuum[idx]
        print("Fraction of nan and inf values: {:.1f}% ({}/{})".format(
            num/n_pixel*100., num, n_pixel))

        # Estimate RMS noise
        print("Estimating noise...")
        noise = cube_helper.estimate_noise(data_line)
        freqs = cube_helper.load_freqs(header, v_LSR)
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
            data_line[idx, cond] = data_contiuum[idx]
        print("Fraction of outliers: {:.1f}% ({}/{})".format(
            num/n_pixel*100., num, n_pixel))

        # Count number of lines
        counts = np.zeros(len(data_contiuum), dtype="i8")
        prominence = noise_factor*noise
        pbar = tqdm(
            enumerate(data_line),
            desc="Counting peaks...",
            total=len(data_line)
        )
        for idx, spec in pbar:
            peaks = signal.find_peaks(
                np.maximum(spec, 0.), # Neglect absoprtion
                prominence=prominence,
                rel_height=rel_height
            )[0]
            counts[idx] += len(peaks)

        # Convert unit
        T_bg = to_kelvin(data_contiuum, freq_m, bmaj, bmin)
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
    beam_data = cube_helper.split_by_mask(
        T_obs_data, T_bg_data, freq_data, noise_data, beam_data, mask_data
    )

    #
    cond = cube_helper.filter_pixel(T_obs_data, noise_data, counts)
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
        fp.attrs["noise_factor"] = float(noise_factor)
        fp.attrs["rel_height"] = rel_height
        fp.attrs["n_row"] = n_row
        fp.attrs["n_col"] = n_col
        fp.attrs["max_count"] = max(counts)
        for key, val in kwargs_filter.items():
            fp.attrs[key] = val
        fp.create_dataset("count", data=counts, dtype="i4")
        fp.create_dataset("index", data=inds_mask, dtype="i4")
        grp_cube = fp.create_group("cube")
        cube_helper.save_file(
            grp_cube, T_obs_data, T_bg_data, freq_data, noise_data, beam_data
        )


@dataclass(frozen=True)
class CubeHelper:
    """Class that provides various functions to process the cube data.

    Args:
        maxiters: Maximum number of iterations used in ``sigma_clipped_stats``.
        n_estimate: Number of pixels to estimate the RMS noise.
        atol_factor: Absolute tolerance factor to remove plateau.
        window_size: Window size to smooth the spectrum when removing plateau.
        f_pla_cut: Plateau fraction below which a pixel is removed.
        noise_factor: RMS noise multipler to identify peaks.
        number_cut: Number of peak below which a pixel is removed.
    """
    maxiters: int = 1000
    n_estimate: int = 10000
    atol_factor: float = 1.e-4
    window_size: int = 31
    noise_factor: float = 4.
    f_pla_cut: float = .75
    number_cut: int = 3

    def prepare_header_list(self, file_list, header_lookup):
        header_list = []
        names = ["freq_start", "dfreq", "n_freq", "BMAJ", "BMIN"]
        for item in file_list:
            header_aux = item.get("header", {})
            header_cont = dict(fits.open(item["contiuum"])[0].header)
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

    def load_freqs(self, header, v_LSR):
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
        freqs *= (1. + const_factor_mu_sigma()[0]*v_LSR)
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
                    height=self.noise_factor*std,
                    prominence=self.noise_factor*std
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

    def save_file(self, grp, T_obs_data, T_bg_data, freq_data, noise_data, beam_data):
        i_seg_save = 0
        for T_obs, T_bg, freqs, noise, beam in zip(
            T_obs_data, T_bg_data, freq_data, noise_data, beam_data
        ):
            grp_sub = grp.create_group(str(i_seg_save))
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

    mat = np.full(shape, np.nan)
    mat[indices[:, 0], indices[:, 1]] = arr
    return mat
