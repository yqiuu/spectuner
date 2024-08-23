import numpy as np
from numba import types, jit
from numba.typed import List, Dict
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
            * planck_radiation(slm_state, nu, T_ex)*term
    return [np.squeeze(spec[:, inds]) for inds in slm_state.slice_list]


def compute_effective_spectra(slm_state, params):
    args = prepare_properties(slm_state, params)
    prop_list = prepare_prop_list(*args)
    freq_list_fine, spec_list_fine = prepare_fine_spectra(
        prop_list, params,
        base_grid=slm_state["base_grid"],
        is_sd_arr=slm_state["is_single_dish"],
        beam_size_sq_arr=slm_state["beam_size_sq"],
        factor_beam_arr=slm_state["factor_beam"],
        factor_freq=slm_state["factor_freq"],
        T_bg_arr=slm_state["T_bg"],
        need_cmb_arr=slm_state["need_cmb"],
        T_cmb=slm_state["T_cmb"]
    )
    freqs_fine = np.concatenate(freq_list_fine)
    spectra_fine = np.concatenate(spec_list_fine)
    return prepare_effective_spectra(slm_state["freq_list"], freqs_fine, spectra_fine)


def prepare_properties(slm_state, params):
    # params (M, 5)
    num = len(params)
    tau_norm_list = [None]*num
    mu_list = [None]*num
    sigma_list = [None]*num
    inds_speice = [None]*num
    inds_segment = [None]*num
    for i_specie, (params_i, sl_data_i) in enumerate(zip(params, slm_state["sl_data"])):
        _, T_ex, den_col, delta_v, v_offset = np.split(params_i, 5, axis=-1)
        tau_norm = np.squeeze(compute_tau_norm(slm_state, sl_data_i, den_col, T_ex))
        tau_norm_list[i_specie] = tau_norm
        mu, sigma = compute_mu_sigma(slm_state, sl_data_i, delta_v, v_offset)
        mu_list[i_specie] = np.squeeze(mu)
        sigma_list[i_specie] = np.squeeze(sigma)
        inds_speice[i_specie] = np.full(len(tau_norm), i_specie)
        inds_segment[i_specie] = sl_data_i["segment"]

    inds_speice_ret = np.concatenate(inds_speice)
    inds_segment_ret = np.concatenate(inds_segment)
    tau_norm_ret = np.concatenate(tau_norm_list)
    mu_ret = np.concatenate(mu_list)
    sigma_ret = np.concatenate(sigma_list)
    left = mu_ret - slm_state["trunc"]*sigma_ret
    right = mu_ret + slm_state["trunc"]*sigma_ret

    inds = np.argsort(left)
    inds_speice_ret = inds_speice_ret[inds]
    inds_segment_ret = inds_segment_ret[inds]
    tau_norm_ret = tau_norm_ret[inds]
    mu_ret = mu_ret[inds]
    sigma_ret = sigma_ret[inds]
    left = left[inds]
    right = right[inds]

    return inds_speice_ret, inds_segment_ret, tau_norm_ret, mu_ret, sigma_ret, left, right


@jit
def prepare_prop_list(inds_specie, inds_segment, tau_norm, mu, sigma, left, right):
    # tau_norm (N,)
    # mu (N,)
    # sigma (N,)
    # Assume left is sorted
    merged = [] # [[left, right], ...]
    prop_list = List()
    for i_specie, i_segment, tau_norm_i, mu_i, sigma_i, left_i, right_i \
        in zip(inds_specie, inds_segment, tau_norm, mu, sigma, left, right):
        if len(merged) == 0 or merged[-1][1] < left_i:
            merged.append([left_i, right_i])
            prop_list.append((
                i_segment, [i_specie], [tau_norm_i], [mu_i], [sigma_i]
            ))
        else:
            merged[-1][1] = max(merged[-1][1], right_i)
            tmp = prop_list[-1]
            tmp[1].append(i_specie)
            tmp[2].append(tau_norm_i)
            tmp[3].append(mu_i)
            tmp[4].append(sigma_i)
    return prop_list


@jit
def prepare_fine_spectra(prop_list, params, base_grid,
                         is_sd_arr, beam_size_sq_arr, factor_beam_arr, factor_freq,
                         T_bg_arr, need_cmb_arr, T_cmb):
    freq_list = []
    spec_list = []
    for i_segment, inds_specie, tau_norm, mu, sigma in prop_list:
        tau_norm = np.asarray(tau_norm)[:, None]
        mu = np.asarray(mu)[:, None]
        sigma = np.asarray(sigma)[:, None]
        nu = np.sort(np.ravel(mu + sigma*base_grid))
        tau = tau_norm*gauss_profile(nu, mu, sigma)

        tmp_dict = Dict.empty(key_type=types.int64, value_type=types.float64[:])
        for i_specie in inds_specie:
            if i_specie not in tmp_dict:
                tmp_dict[i_specie] = np.zeros_like(nu)
        for i_specie, tau_i in zip(inds_specie, tau):
            tmp_dict[i_specie] += tau_i

        theta = params[:, :1]
        T_ex = params[:, 1:2]
        is_single_dish = is_sd_arr[i_segment]
        beam_size_sq = beam_size_sq_arr[i_segment]
        factor_beam = factor_beam_arr[i_segment]
        T_bg = T_bg_arr[i_segment]
        need_cmb = need_cmb_arr[i_segment]

        spec = np.zeros_like(nu)
        for i_specie, tau_total in tmp_dict.items():
            theta_i = theta[i_specie]
            T_ex_i = T_ex[i_specie]
            spec_i = planck_radiation(nu, T_ex_i, factor_freq) - T_bg
            if need_cmb:
                spec_i -= planck_radiation(nu, T_cmb, factor_freq)
            spec_i *= compute_filling_factor(nu, theta_i, is_single_dish, beam_size_sq, factor_beam)
            spec_i *= 1 - np.exp(-tau_total)
            spec += spec_i

        freq_list.append(nu)
        spec_list.append(spec)
    return freq_list, spec_list


@jit
def prepare_effective_spectra(freq_list, freqs_fine, spectra_fine):
    spec_list = []
    for freqs in freq_list:
        freqs_p = np.zeros(len(freqs) + 1)
        freqs_p[1:-1] = .5*(freqs[1:] + freqs[:-1])
        freqs_p[0] = freqs[0] - .5*(freqs[1] - freqs[0])
        freqs_p[-1] = freqs[-1] + .5*(freqs[-1] - freqs[-2])
        spectra_p = np.interp(freqs_p, freqs_fine, spectra_fine)
        inds = np.searchsorted(freqs_fine, freqs_p)
        spec_eff = [0.]*len(freqs)
        for i_freq in range(len(freqs)):
            idx_b = inds[i_freq]
            idx_e = inds[i_freq + 1]
            x_freq = np.zeros(idx_e - idx_b + 2)
            x_freq[1:-1] = freqs_fine[idx_b:idx_e]
            x_freq[0] = freqs_p[i_freq]
            x_freq[-1] = freqs_p[i_freq + 1]
            y_spec = np.zeros(idx_e - idx_b + 2)
            y_spec[1:-1] = spectra_fine[idx_b:idx_e]
            y_spec[0] = spectra_p[i_freq]
            y_spec[-1] =  spectra_p[i_freq + 1]
            spec_eff[i_freq] = np.trapz(y_spec, x_freq)
        spec_list.append(np.asarray(spec_eff)/np.diff(freqs_p))
    return spec_list


def compute_tau_total(slm_state, sl_data, den_col, T_ex, delta_v, v_offset):
    # den_col (M, 1) cm^-2
    # T_ex (M, 1) K
    # v_offset (M,) km/s
    # delta_v (M,) km/s
    mu, sigma = compute_mu_sigma(slm_state, sl_data, delta_v, v_offset)
    mu = mu[:, None]
    sigma = sigma[:, None]
    nu = slm_state["freqs"] # (N_nu,)
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
    E_trans = slm_state["factor_freq"]*sl_data["freq"]
    return slm_state["factor_tau"]*den_col*sl_data["A_ul"]*sl_data["g_u"] \
        / (Q_T*sl_data["freq"]*sl_data["freq"]) \
        * np.exp(-sl_data["E_low"]/T_ex)*(1 - np.exp(-E_trans/T_ex))


def compute_mu_sigma(slm_state, sl_data, delta_v, v_offset):
    # sl_data (N_t,)
    # delta_v (B, 1)
    # v_offset (B, 1)
    mu = (1 - slm_state["factor_v_offset"]*v_offset)*sl_data["freq"] # (M, N_t)
    sigma = slm_state["factor_delta_v"]*delta_v*mu
    return mu, sigma

@jit
def compute_filling_factor(nu, theta, is_single_dish, beam_size_sq, factor_beam):
    # nu (N,)
    # theta (B, 1)
    # Return (B, 1)
    theta_sq = theta*theta
    if is_single_dish:
        beam_size_sq_ = np.square(factor_beam/nu)
    else:
        beam_size_sq_ = np.full_like(nu, beam_size_sq)
    return theta_sq/(beam_size_sq_ + theta_sq)

@jit
def planck_radiation(nu, T_ex, factor_freq):
    # nu (N,)
    # T_ex (M, 1)
    # Return (M, N)
    return factor_freq*nu/(np.exp(factor_freq*nu/T_ex) - 1)

@jit
def gauss_profile(x, mu, sigma):
    # x (N,)
    # mu (M, 1)
    # sigma (M, 1)
    # Return (M, N)
    return np.exp(-.5*np.square((x - mu)/sigma))/(np.sqrt(2*np.pi)*sigma)


def derive_base_grid(trunc, eps):
    def trapz_rule(f, a, b):
        return .5*(b - a)*(f(a) + f(b))

    def adaptive_trapz(f, a, b, points, eps):
        mid = .5*(a + b)
        val_0 = trapz_rule(f, a, b)
        val_1 = trapz_rule(f, a, mid) + trapz_rule(f, mid, b)
        points.append(mid)

        if abs(val_0 - val_1) < eps:
            return val_1
        else:
            return adaptive_trapz(f, a, mid, points, .5*eps) \
                + adaptive_trapz(f, mid, b, points, .5*eps)

    def func(x):
        return np.exp(-.5*x*x)/np.sqrt(2*np.pi)

    points = [-trunc, trunc]
    adaptive_trapz(func, -trunc, trunc, points, eps)
    points.sort()
    points = np.asarray(points)
    return points


def create_spectral_line_model_state(sl_data, freq_list, obs_info, trunc=10., eps_grid=1e-3):
    """
    Args:
        obs_info:
            - beam_info (float or tuple): Telescople size in meter or
            (BMAJ, BMIN) in degree.
            - T_bg (float): background temperature.
            - need_cmb (bool): Whethter to add additional CMB radiation in
            the continuum.
    """
    assert len(freq_list) == len(obs_info)

    slm_state = {}
    #
    slm_state["sl_data"] = sl_data
    slm_state["freq_list"] = freq_list
    slm_state["freqs"] = np.concatenate(freq_list)
    #
    slice_list = []
    idx_b = 0
    for freq in freq_list:
        slice_list.append(slice(idx_b, idx_b + len(freq)))
        idx_b += len(freq)
    slm_state["slice_list"] = slice_list
    #
    slm_state["factor_theta"] = 1.22*3600*180/np.pi*(constants.c/units.MHz).to(units.m).value
    slm_state["factor_v_offset"] = 1./(constants.c).to(units.km/units.second).value
    slm_state["factor_delta_v"] = slm_state["factor_v_offset"]/(2*np.sqrt(2*np.log(2)))
    slm_state["factor_tau"] = ((constants.c/units.MHz)**2/units.second).to(units.cm**2*units.MHz).value/(8*np.pi)
    slm_state["factor_freq"] = (constants.h/constants.k_B*units.MHz).to(units.Kelvin).value
    # Set beam info
    is_single_dish = []
    factor_beam = []
    beam_size_sq = []
    for info_dict in obs_info:
        beam_info = info_dict["beam_info"]
        if np.isscalar(beam_info):
            # Single dish
            is_single_dish.append(True)
            factor_beam.append(slm_state["factor_theta"]/beam_info)
            beam_size_sq.append(0.)
        else:
            # Interferometery
            is_single_dish.append(False)
            factor_beam.append(1.)
            # Convert deg to arcsecond
            beam_size_sq.append(beam_info[0]*beam_info[1]*3600*3600)
    slm_state["is_single_dish"] = is_single_dish
    slm_state["factor_beam"] = factor_beam
    slm_state["beam_size_sq"] = beam_size_sq
    #
    T_bg_arr = []
    need_cmb = []
    for info_dict in obs_info:
        T_bg_arr.append(info_dict["T_bg"])
        need_cmb.append(info_dict["need_cmb"])
    slm_state["T_bg"] = T_bg_arr
    slm_state["need_cmb"] = need_cmb
    slm_state["T_cmb"] = np.array([2.726]) # K
    #
    slm_state["trunc"] = trunc
    slm_state["base_grid"] = derive_base_grid(trunc, eps_grid)
    #
    return slm_state