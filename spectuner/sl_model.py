import numpy as np
from numba import jit
from astropy import constants, units


def compute_spectra_simple(slm_state, params):
    # params (B, 5)
    spec = 0.
    for params_i, sl_data_i in zip(params, slm_state.sl_data):
        theta, T_ex, den_col, delta_v, v_offset = np.split(params_i, 5, axis=-1)
        tau_total = compute_tau_total(slm_state, sl_data_i, den_col, T_ex, delta_v, v_offset)
        term = 1 - np.exp(-tau_total)
        nu = slm_state.freqs
        spec += compute_filling_factor(slm_state, nu, theta) \
            * planck_profile(slm_state, nu, T_ex)*term
    return [np.squeeze(spec[:, inds]) for inds in slm_state.slice_list]


def compute_effective_spectra(slm_state, params):
    inds_specie, tau_norm, mu, sigma =prepare_properties(slm_state, params)
    prop_list = prepare_prop_list(slm_state, inds_specie, tau_norm, mu, sigma)
    freqs_fine, spectra_fine = process_prop_list(slm_state, prop_list, params)
    return prepare_effective_spectra(slm_state.freq_list, freqs_fine, spectra_fine)


def prepare_properties(slm_state, params):
    # params (M, 5)
    num = len(params)
    tau_norm_list = [None]*num
    mu_list = [None]*num
    sigma_list = [None]*num
    inds_list = [None]*num
    for i_specie, (params_i, sl_data_i) in enumerate(zip(params, slm_state.sl_data)):
        _, T_ex, den_col, delta_v, v_offset = np.split(params_i, 5, axis=-1)
        tau_norm = np.squeeze(compute_tau_norm(slm_state, sl_data_i, den_col, T_ex))
        tau_norm_list[i_specie] = tau_norm
        mu, sigma = compute_mu_sigma(slm_state, sl_data_i, delta_v, v_offset)
        mu_list[i_specie] = np.squeeze(mu)
        sigma_list[i_specie] = np.squeeze(sigma)
        inds_list[i_specie] = np.full(len(tau_norm), i_specie)

    inds_speice = np.concatenate(inds_list)
    tau_norm_ret = np.concatenate(tau_norm_list)
    mu_ret = np.concatenate(mu_list)
    sigma_ret = np.concatenate(sigma_list)

    inds = np.argsort(mu_ret)
    inds_speice = inds_speice[inds]
    tau_norm_ret = tau_norm_ret[inds]
    mu_ret = mu_ret[inds]
    sigma_ret = sigma_ret[inds]

    return inds_speice, tau_norm_ret, mu_ret, sigma_ret


def prepare_prop_list(slm_state, inds_specie, tau_norm, mu, sigma):
    # tau_norm (N,)
    # mu (N,)
    # sigma (N,)
    # Assume mu is sorted
    left = mu - slm_state.trunc*sigma
    right = mu + slm_state.trunc*sigma

    merged = [] # [[left, right], ...]
    prop_list = []
    for i_specie, tau_norm_i, mu_i, sigma_i, left_i, right_i \
        in zip(inds_specie, tau_norm, mu, sigma, left, right):
        if len(merged) == 0 or merged[-1][1] < left_i:
            merged.append([left_i, right_i])
            prop_list.append((
                [i_specie], [tau_norm_i], [mu_i], [sigma_i]
            ))
        else:
            merged[-1][1] = max(merged[-1][1], right_i)
            tmp = prop_list[-1]
            tmp[0].append(i_specie)
            tmp[1].append(tau_norm_i)
            tmp[2].append(mu_i)
            tmp[3].append(sigma_i)
    return prop_list


def process_prop_list(slm_state, prop_list, params):
    freq_list = []
    spec_list = []
    for inds_specie, tau_norm, mu, sigma in prop_list:
        tau_norm = np.asarray(tau_norm)[:, None]
        mu = np.asarray(mu)[:, None]
        sigma = np.asarray(sigma)[:, None]
        nu = mu + slm_state.trunc*sigma*slm_state.base_grid
        nu = np.sort(np.ravel(nu))
        tau = tau_norm*gauss_profile(nu, mu, sigma)

        tmp_dict = {idx: 0. for idx in inds_specie}
        for idx, tau_i in zip(inds_specie, tau):
            tmp_dict[idx] += tau_i

        theta, T_ex, *_ = np.split(params, 5, axis=-1)
        spec = 0.
        for idx, tau_total in tmp_dict.items():
            theta_i = theta[idx]
            T_ex_i = T_ex[idx]
            spec += compute_filling_factor(slm_state, nu, theta_i) \
                * planck_profile(slm_state, nu, T_ex_i)*(1 - np.exp(-tau_total))

        freq_list.append(nu)
        spec_list.append(spec)
    freqs = np.concatenate(freq_list)
    spectra = np.concatenate(spec_list)
    return freqs, spectra


@jit
def prepare_effective_spectra(freq_list, freqs_fine, spectra_fine):
    spec_list = []
    for freqs in freq_list:
        freqs_p = np.zeros(len(freqs) + 1)
        freqs_p[1:-1] = .5*(freqs[1:] + freqs[:-1])
        freqs_p[0] = freqs[0] - .5*(freqs[1] - freqs[0])
        freqs_p[-1] = freqs[-1] + .5*(freqs[-1] - freqs[-2])
        spec_eff = [0.]*len(freqs)
        for i_freq, (lower, upper) in enumerate(zip(freqs_p[:-1], freqs_p[1:])):
            idx_b, idx_e = np.searchsorted(freqs_fine, [lower, upper])
            x_freq = np.zeros(idx_e - idx_b + 2)
            x_freq[1:-1] = freqs_fine[idx_b:idx_e]
            x_freq[0] = lower
            x_freq[-1] = upper
            y_spec = np.zeros(idx_e - idx_b + 2)
            y_spec[1:-1] = spectra_fine[idx_b:idx_e]
            y_lower, y_upper = np.interp([lower, upper], freqs_fine, spectra_fine)
            y_spec[0] = y_lower
            y_spec[-1] = y_upper
            spec_eff[i_freq] = np.trapz(y_spec, x_freq)/(upper - lower)
        spec_list.append(np.asarray(spec_eff))
    return spec_list


def compute_tau_total(slm_state, sl_data, den_col, T_ex, delta_v, v_offset):
    # den_col (M, 1) cm^-2
    # T_ex (M, 1) K
    # v_offset (M,) km/s
    # delta_v (M,) km/s
    mu, sigma = compute_mu_sigma(slm_state, sl_data, delta_v, v_offset)
    mu = mu[:, None]
    sigma = sigma[:, None]
    nu = slm_state.freqs # (N_nu,)
    phi = np.exp(-.5*np.square((nu - mu)/sigma))/(np.sqrt(2*np.pi)*sigma)
    tau = compute_tau_norm(slm_state, sl_data, den_col, T_ex)[..., None]*phi # (M, N_t, N_nu)
    tau_total = np.sum(tau, axis=-2)
    return tau_total


def compute_tau_norm(slm_state, sl_data, den_col, T_ex):
    # sl_data (N_t,)
    # den (B, 1)
    # T_ex (B, 1)
    # Return (B, N_t)
    Q_T = np.interp(np.ravel(T_ex), sl_data["x_T"], sl_data["Q_T"])[:, None]
    E_trans = slm_state.factor_freq*sl_data["freq"]
    return slm_state.factor_tau*den_col*sl_data["A_ul"]*sl_data["g_u"] \
        / (Q_T*sl_data["freq"]*sl_data["freq"]) \
        * np.exp(-sl_data["E_low"]/T_ex)*(1 - np.exp(-E_trans/T_ex))


def compute_mu_sigma(slm_state, sl_data, delta_v, v_offset):
    # sl_data (N_t,)
    # delta_v (B, 1)
    # v_offset (B, 1)
    mu = (1 - slm_state.factor_v_offset*v_offset)*sl_data["freq"] # (M, N_t)
    sigma = slm_state.factor_delta_v*delta_v*mu
    return mu, sigma


def compute_filling_factor(slm_state, nu, theta):
    # nu (N,)
    # theta (B, 1)
    # Return (B, 1)
    theta_sq = theta*theta
    if slm_state.factor_beam is None:
        # Interferometery
        beam_size_sq = slm_state.beam_size_sq
    else:
        # Single dish
        beam_size_sq = np.square(slm_state.factor_beam/nu)
    return theta_sq/(beam_size_sq + theta_sq)


def planck_profile(slm_state, nu, T_ex):
    # nu (N,)
    # T_ex (M, 1)
    # Return (M, N)
    return slm_state.factor_freq*nu/(np.exp(slm_state.factor_freq*nu/T_ex) - 1)


def gauss_profile(x, mu, sigma):
    # x (N,)
    # mu (M, 1)
    # sigma (M, 1)
    # Return (M, N)
    return np.exp(-.5*np.square((x - mu)/sigma))/(np.sqrt(2*np.pi)*sigma)


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
            self.factor_beam = self.factor_theta/beam_info
            self.beam_size_sq = None
        else:
            # Convert deg to arcsecond
            self.factor_beam = None
            self.beam_size_sq = beam_info[0]*beam_info[1]*3600*3600
        #
        self.trunc = 10.
        self.base_grid = np.linspace(-1, 1, 11)