import yaml
import math
from functools import lru_cache

import numpy as np
import spectuner
from scipy.signal import find_peaks
from astropy import units, constants

from ..sl_model import create_spectral_line_db


def create_embeding_model(config_embed, sl_db=None):
    if sl_db is None:
        sl_db = create_spectral_line_db(
            config_embed["fname"], config_embed.get("cache", False)
        )
    kwargs = {
        "sl_db": sl_db,
        "norms_sl": config_embed["norms_sl"],
        "v_width": config_embed["v_width"],
        "n_grid": config_embed["n_grid"],
        "max_length": config_embed["max_length"],
    }
    name = config_embed.get("name", "v1")
    if name == "v3":
        embedding_model = EmbeddingV3(**kwargs)
    else:
        raise ValueError(f"Unknown embedding model: {name}.")
    return embedding_model


def standard_norm(x, mu, std, x_min, x_max, scale):
    if x_min is None and x_max is None:
        x_out = x
    else:
        x_out = np.clip(x, x_min, x_max)

    if scale == "log":
        x_out = np.log10(x_out)
    elif scale == "arcsinh":
        x_out = np.arcsinh(x_out)
    elif scale == "linear":
        pass
    else:
        raise ValueError(f"Unknown scale: {scale}")

    return (x_out - mu)/std


@lru_cache(maxsize=None)
def const_beam() -> float:
    return 1.22*180/np.pi*(constants.c/units.MHz).to(units.m).value


def diameter_to_beam_size(diameter: float, freq: np.ndarray) -> np.ndarray:
    """Convert telescope diameter to beam size

    Args:
        diameter: telescope diameter (m).
        freq: frequency (MHz).

    Returns:
        beam_size (degree).
    """
    return const_beam()/(diameter*freq)


def beam_size_to_diameter(beam_info: tuple, freq_c: float) -> float:
    """Convert beam size to telescope diameter.

    Args:
        beam_info: (beam_size, beam_size) (degree).
        freq_c: frequency (MHz).

    Returns:
        Telescope diameter (m).
    """
    beam_size = math.sqrt(beam_info[0]*beam_info[1])
    return const_beam()/(beam_size*freq_c)


class EmbeddingV3:
    def __init__(self, sl_db, norms_sl, v_width=150., n_grid=128,
                 max_length=None):
        self.sl_db = sl_db
        self.norms_sl = yaml.safe_load(open(norms_sl))
        self.v_width = v_width
        self.n_grid = n_grid
        self.max_length = max_length

    def __call__(self, obs_info, specie):
        freq_data = [item["spec"][:, 0] for item in obs_info]
        sl_dict = self.sl_db.query_sl_dict(specie, freq_data)
        if len(sl_dict["freq"]) == 0:
            return

        if self.max_length is not None and len(sl_dict["freq"]) > self.max_length:
            sl_dict = self.random_pick(sl_dict, self.max_length)

        # Create embed_sl
        embed_sl = np.vstack([
            standard_norm(sl_dict["A_ul"], **self.norms_sl["A_ul"]),
            standard_norm(sl_dict["g_u"], **self.norms_sl["g_u"]),
            standard_norm(sl_dict["E_low"],  **self.norms_sl["E_low"]),
            standard_norm(sl_dict["E_up"], **self.norms_sl["E_up"])
        ]).T
        q_t = standard_norm(sl_dict["Q_T"], **self.norms_sl["Q_T"])
        embed_sl = np.hstack([embed_sl, np.tile(q_t, (len(embed_sl), 1))])
        embed_sl = embed_sl.astype("f4")

        # Create embed_obs
        embed_obs = []
        for i_segment, freq_c in zip(sl_dict["segment"], sl_dict["freq"]):
            item = obs_info[i_segment]
            freq, T_obs = item["spec"].T
            T_bg = item["T_bg"]
            noise = item["noise"]
            beam_info = item["beam_info"]
            embed_obs.append(self.create_patch(
                freq_c, freq, T_obs, T_bg, noise, beam_info,
                self.v_width, self.n_grid
            ))
        embed_obs = np.stack(embed_obs, dtype="f4")

        specie_list = [{"id": 0, "root": specie, "species": [specie]}]
        return embed_obs, embed_sl, sl_dict, specie_list

    def create_patch(self, freq_c, freq, spec, T_bg, noise, beam_info, v_width, n_grid):
        # Find peaks
        inds, _ = find_peaks(spec, prominence=4*noise)
        peaks = np.zeros_like(spec)
        peaks[inds] = 1.

        freq_min = spectuner.compute_shift(freq_c, -v_width)
        freq_max = spectuner.compute_shift(freq_c, v_width)
        freq_patch = np.linspace(freq_min, freq_max, n_grid)
        inds_left = np.searchsorted(freq, freq_patch) - 1
        inds_left[inds_left < 0] = 0
        inds_rihgt = inds_left + 1
        inds_rihgt[inds_rihgt >= len(freq)] = len(freq) - 1

        delta = freq[inds_left] - freq[inds_rihgt]
        cond = delta == 0.
        delta[cond] = 1.
        frac = (freq_patch - freq[inds_rihgt])/delta
        frac[cond] = 0.
        r_dfreq = (freq_patch[1] - freq_patch[0])/np.mean(np.diff(freq))
        r_dfreq = np.full_like(freq_patch, r_dfreq)

        spec_patch_left = np.arcsinh(spec[inds_left])
        spec_patch_right = np.arcsinh(spec[inds_rihgt])
        peaks_left = peaks[inds_left]
        peaks_right = peaks[inds_rihgt]

        # Check bounds
        bounds = (freq_patch >= freq[0]) & (freq_patch <= freq[-1])

        # Compute velocity shifts
        v_diff = (freq_patch/freq_c - 1)/spectuner.const_factor_mu_sigma()[0]
        v_diff /= v_width

        #
        T_bg_patch = np.full_like(freq_patch, np.arcsinh(T_bg))
        noise_patch = np.full_like(freq_patch, np.arcsinh(noise))

        # Compute beam info
        beam_scaled = self.compute_scaled_beam(beam_info, freq_patch)

        return np.stack([
            v_diff, frac, spec_patch_left, spec_patch_right,
            peaks_left, peaks_right, bounds,
            r_dfreq, T_bg_patch, noise_patch, beam_scaled,
        ])

    def random_pick(self, sl_dict, tgt_length):
        sl_dict_ret = {}
        inds = np.random.choice(len(sl_dict["freq"]), tgt_length, replace=False)
        for key, arr in sl_dict.items():
            if key != "Q_T" and key != "x_T":
                arr = arr[inds]
            sl_dict_ret[key] = arr # No copy
        return sl_dict_ret

    def compute_scaled_beam(self, beam_info, freq):
        if isinstance(beam_info, float):
            beam_size = diameter_to_beam_size(beam_info, freq)
        else:
            beam_size = np.full_like(freq, math.sqrt(beam_info[0]*beam_info[1]))
        log_bs_min = -2
        log_bs_max = 1.
        beam_scaled = (np.log10(beam_size) - log_bs_min)/(log_bs_max - log_bs_min)
        return beam_scaled