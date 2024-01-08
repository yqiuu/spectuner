import os
import sys
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np

# Import XCLASS
from xclass import task_myXCLASS, task_ListDatabase


def create_molfit_file(fname, mol_names, params, include_list):
    text = f"""% name of molecule		number of components
	% source size [arcsec]:    T_rot [K]:    N_tot [cm-2]:    velocity width [km/s]: velocity offset [km/s]:    CFflag:\n"""
    for name, (source_size, T_rot, N_tot, vel_width, vel_offset) in zip(mol_names, params):
        if include_list is None or name in include_list:
            text += f"{name}   1\n"
            text += f"{source_size}    {T_rot}     {N_tot}     {vel_width}     {vel_offset}   c\n"
    open(fname, mode='w').write(text)


def load_molfit_info(fname):
    # mol_names (N,)
    # bounds (N, X, 2)
    mol_names = []
    bounds = []
    for line in open(fname).readlines():
        line = line.strip()
        if line.startswith("%"):
            continue

        if line.startswith("y"):
            line = line.split()
            bounds.append(np.array([
                [float(line[1]), float(line[2])],
                [float(line[5]), float(line[6])],
                [float(line[9]), float(line[10])],
                [float(line[13]), float(line[14])],
                [float(line[17]), float(line[18])],
            ]))
        else:
            mol_names.append(line.split()[0])
    bounds = np.vstack(bounds).reshape(len(mol_names), -1, 2)
    return mol_names, bounds


def create_wrapper_from_config(obs_data, mol_dict, config, **kwargs):
    if isinstance(obs_data, np.ndarray):
        freq_min, freq_max, freq_step = derive_freq_range(obs_data)
    else:
        freq_min, freq_max, freq_step = 0., 1., .1

    wrapper = XCLASSWrapper(
        FreqMin=freq_min,
        FreqMax=freq_max,
        FreqStep=freq_step,
        mol_dict=mol_dict,
        **config,
        **kwargs
    )
    return wrapper


def derive_freq_range(freq):
    freq_step = freq[1] - freq[0]
    freq_min = freq[0]
    freq_max = freq[-1] + .1*freq_step
    return freq_min, freq_max, freq_step


def extract_line_frequency(transitions):
    """Extract line frequency with Doppler-shift.

    Args:
        transitions (list): Transitions obtained from XCLASS.

    Returns:
        dict: Each item uses the molecular name as the key and gives a list of
            transition frequencies and .
    """
    trans_dict = defaultdict(list)
    for item in transitions[1:]:
        trans_dict[item[-1]].append(float(item[1]))
    return trans_dict


class XCLASSWrapper:
    def __init__(self, mol_dict, prefix_molfit,
                 FreqMin=0., FreqMax=1., FreqStep=.1,
                 IsoTableFileName=None, **xclass_kwargs):
        if IsoTableFileName is None:
            xclass_kwargs["iso_flag"] = False
        else:
            xclass_kwargs["iso_flag"] = True
        xclass_kwargs["IsoTableFileName"] = IsoTableFileName
        self.update_frequency(FreqMin, FreqMax, FreqStep)

        misc_names = []
        def set_misc_var(var_name):
            if not var_name in xclass_kwargs:
                misc_names.append(var_name)

        set_misc_var("tBack")
        set_misc_var("tSlope")
        set_misc_var("vLSR")
        self._xclass_kwargs = xclass_kwargs

        n_param_per_mol = 5
        idx_den = 2
        self.pm = ParameterManager(
            mol_dict, n_param_per_mol, idx_den, misc_names
        )
        self.prefix_molfit = prefix_molfit
        self.include_list = None

    def update_frequency(self, freq_min, freq_max, freq_step):
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.freq_step = freq_step

    def update_include_list(self, include_list):
        self.include_list = include_list

    def call(self, params, remove_dir=True):
        mol_names, params_mol, params_dict = self.pm.derive_params(params)
        spec, log, trans, tau, job_dir = self.call_params_dict(
            mol_names, params_mol, params_dict, remove_dir
        )
        if len(spec) == 0:
            spec = None
        else:
            spec = spec[:, 1]
        return spec, log, trans, tau, job_dir

    def call_params_dict(self, mol_names, params_mol, params_dict, remove_dir):
        fname_molfit = "{}_{}.molfit".format(self.prefix_molfit, os.getpid())
        create_molfit_file(fname_molfit, mol_names, params_mol, self.include_list)

        spec, log, trans, tau, job_dir = task_myXCLASS.myXCLASSCore(
            FreqMin=self.freq_min,
            FreqMax=self.freq_max,
            FreqStep=self.freq_step,
            MolfitsFileName=fname_molfit,
            **params_dict,
            **self._xclass_kwargs
        )
        if remove_dir:
            shutil.rmtree(job_dir)
        return spec, log, trans, tau, job_dir

    def create_molfit_file(self, fname, params):
        mol_names, params_mol, _ = self.pm.derive_params(params)
        create_molfit_file(fname, mol_names, params_mol, self.include_list)


class ParameterManager:
    def __init__(self, mol_dict, n_param_per_mol, idx_den, misc_names):
        # Set indices
        idx = 0
        n_mol_param = len(mol_dict)*n_param_per_mol
        self.inds_mol_param = slice(idx, n_mol_param)
        idx += n_mol_param

        n_iso_param = 0
        for names in mol_dict.values():
            n_iso_param += len(names)
        self.inds_iso_param = slice(idx, idx + n_iso_param)
        idx += n_iso_param

        self.inds_misc_param = slice(idx, idx + len(misc_names))
        self.n_tot_param = idx + len(misc_names)

        #
        iso_inds = {}
        idx_b = 0
        for name, inds in mol_dict.items():
            if len(inds) == 0:
                continue

            idx_e = idx_b + len(inds)
            iso_inds[name] = slice(idx_b, idx_e)
            idx_b = idx_e
        self.iso_inds = iso_inds

        #/
        self.mol_dict = mol_dict
        self.n_mol = len(mol_dict)
        self.n_mol_param = n_mol_param
        self.n_iso_param = n_iso_param
        self.n_param_per_mol = n_param_per_mol

        self.idx_den = idx_den
        self.misc_names = misc_names

    def derive_params(self, params):
        if len(params) != self.n_tot_param:
            raise ValueError(f"Total number of parameters should be {self.n_tot_param}.")

        mol_names, params_mol = self.derive_mol_params(params)
        params_misc = params[self.inds_misc_param]
        params_dict = {}
        for key, val in zip(self.misc_names, params_misc):
            params_dict[key] = val
        return mol_names, params_mol, params_dict

    def derive_mol_params(self, params):
        # params (N,)
        params_mol = params[self.inds_mol_param].reshape(-1, self.n_param_per_mol)
        params_iso = params[self.inds_iso_param]

        mol_names = []
        params_mol_ret = []
        for idx, (name, iso_list) in enumerate(self.mol_dict.items()):
            mol_names.append(name)
            params_mol_ret.append(params_mol[idx])
            idx_iso = 0
            for name_iso in iso_list:
                mol_names.append(name_iso)
                params_tmp = params_mol[idx].copy()
                params_tmp[self.idx_den] = params_iso[idx_iso]
                params_mol_ret.append(params_tmp)
                idx_iso += 1
        params_mol_ret = np.vstack(params_mol_ret)
        return mol_names, params_mol_ret

    def get_mol_slice(self, mol_name):
        idx = list(self.mol_dict.keys()).index(mol_name)
        return slice(idx*self.n_param_per_mol, (idx + 1)*self.n_param_per_mol)

    def get_iso_slice(self, mol_name):
        if mol_name in self.iso_inds:
            return self.iso_inds[mol_name]

    def get_misc_params(self, key, params):
        params_misc = params[self.inds_misc_param]
        idx = self.misc_names.index(key)
        return params_misc[idx]