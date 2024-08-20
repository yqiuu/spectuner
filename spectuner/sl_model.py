import numpy as np
from astropy import constants, units


def compute_spectra(slm_state, params):
    # params (B, 5)
    spec = 0.
    for params_i, sl_data_i in zip(params, slm_state.sl_data):
        theta, T_ex, den_col, delta_v, v_offset = np.split(params_i, 5, axis=-1)
        tau_total = compute_tau_total(slm_state, sl_data_i, den_col, T_ex, delta_v, v_offset)
        term = 1 - np.exp(-tau_total)
        spec += compute_filling_factor(slm_state, theta)*planck_profile(slm_state, T_ex)*term
    return [np.squeeze(spec[:, inds]) for inds in slm_state.slice_list]


def compute_tau_total(slm_state, sl_data, den_col, T_ex, delta_v, v_offset):
    # den_col (B, 1) cm^-2
    # T_ex (B, 1) K
    # v_offset (B,) km/s
    # delta_v (B,) km/s
    nu_c = (1 - slm_state.factor_v_offset*v_offset)*sl_data["freq"] # (B, N_t)
    nu_c = nu_c[..., None] # (B, N_t, 1)
    nu = slm_state.freqs # (N_nu,)
    sigma = slm_state.factor_delta_v*delta_v[:, None]*nu_c # (B, N_t, 1)
    phi = np.exp(-.5*np.square((nu - nu_c)/sigma))/(np.sqrt(2*np.pi)*sigma)
    tau = compute_tau_max(slm_state, sl_data, den_col, T_ex)[..., None]*phi # (B, N_t, N_nu)
    tau_total = np.sum(tau, axis=-2)
    return tau_total


def compute_tau_max(slm_state, sl_data, den_col, T_ex):
    # den (B, 1)
    # T_ex (B, 1)
    # sl_data (N,) or (B, N_t)
    # Return (B, N_t)
    Q_T = np.interp(np.ravel(T_ex), sl_data["x_T"], sl_data["Q_T"])[:, None]
    E_trans = slm_state.factor_freq*sl_data["freq"]
    return slm_state.factor_tau*den_col*sl_data["A_ul"]*sl_data["g_u"] \
        / (Q_T*sl_data["freq"]*sl_data["freq"]) \
        * np.exp(-sl_data["E_low"]/T_ex)*(1 - np.exp(-E_trans/T_ex))


def compute_filling_factor(slm_state, theta):
    # theta (B, 1)
    # Return (B, 1)
    theta_sq = theta*theta
    return theta_sq/(slm_state.beam_size_sq + theta_sq)


def planck_profile(slm_state, T_ex):
    nu = slm_state.freqs
    return slm_state.factor_freq*nu/(np.exp(slm_state.factor_freq*nu/T_ex) - 1)


class SpectralLineModelState:
    def __init__(self, sl_data, freq_list, beam_info):
        self.sl_data = sl_data
        self.freq_list = freq_list
        self.freqs = np.concatenate(freq_list)
        #
        slice_list = []
        idx_b = 0
        for freq in freq_list:
            slice_list.append(slice(idx_b, idx_b + len(freq)))
            idx_b += len(freq)
        self.slice_list = slice_list
        # Set constants
        self.factor_theta = 1.22*3600*180/np.pi*(constants.c/units.MHz).to(units.m).value
        self.factor_v_offset = 1./(constants.c).to(units.km/units.second).value
        self.factor_delta_v = self.factor_v_offset/(2*np.sqrt(2*np.log(2)))
        self.factor_tau = ((constants.c/units.MHz)**2/units.second).to(units.cm**2*units.MHz).value/(8*np.pi)
        self.factor_freq = (constants.h/constants.k_B*units.MHz).to(units.Kelvin).value
        #
        if isinstance(beam_info, float) or isinstance(beam_info, int):
            # Single dish
            beam_size = self.factor_theta/(beam_info*self.freqs)
            self.beam_size_sq = beam_size*beam_size
        else:
            # Convert deg to arcsecond
            self.beam_size_sq = beam_info[0]*beam_info[1]*3600*3600