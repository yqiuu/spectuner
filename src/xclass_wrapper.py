import os
import sys
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np

# Import XCLASS
path = os.environ.get("XCLASSRootDir")
if path is None:
    raise ValueError("Set environment variable 'XCLASSRootDir'.")
sys.path.append(str(Path(path)/Path("build_tasks")))
import task_myXCLASS
import task_ListDatabase


def create_molfit_file(fname, mol_names, params):
    text = f"""% name of molecule		number of components
	% source size [arcsec]:    T_rot [K]:    N_tot [cm-2]:    velocity width [km/s]: velocity offset [km/s]:    CFflag:\n"""
    for name, (source_size, T_rot, N_tot, vel_width, vel_offset) in zip(mol_names, params):
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


def create_wrapper_from_config(spec_obs, mol_dict, config, **kwargs):
    freq = spec_obs[:, 0].copy()
    freq_min = freq[0]
    freq_max = freq[-1]
    freq_step = freq[1] - freq[0]

    wrapper = XCLASSWrapper(
        FreqMin=freq_min,
        FreqMax=freq_max,
        FreqStep=freq_step,
        mol_dict=mol_dict,
        **config,
        **kwargs
    )
    return wrapper


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
    def __init__(self, FreqMin, FreqMax, FreqStep, TelescopeSize, inter_flag,
                 RestFreq, nH_flag, N_H, kappa_1300, beta_dust, IsoTableFileName,
                 mol_dict, prefix_molfit,
                 t_back_flag=True, tBack=None, tslope=None, vLSR=None):
        xclass_kwargs = {
            "FreqMin": FreqMin,
            "FreqMax": FreqMax,
            "FreqStep": FreqStep,
            "TelescopeSize": TelescopeSize,
            "Inter_Flag": inter_flag,
            "RestFreq": RestFreq,
            "nH_flag": nH_flag,
            "N_H": N_H,
            "kappa_1300": kappa_1300,
            "beta_dust": beta_dust,
            "t_back_flag": t_back_flag
        }
        if IsoTableFileName is None:
            xclass_kwargs["iso_flag"] = False
        else:
            xclass_kwargs["iso_flag"] = True
        xclass_kwargs["IsoTableFileName"] = IsoTableFileName
        self._xclass_kwargs = xclass_kwargs

        misc_names = []
        def set_misc_var(var, var_name):
            if var is None:
                misc_names.append(var_name)
            else:
                xclass_kwargs[var_name] = var

        set_misc_var(tBack, "tBack")
        set_misc_var(tslope, "tslope")
        set_misc_var(vLSR, "vLSR")

        n_param_per_mol = 5
        idx_den = 2
        self.pm = ParameterManager(
            mol_dict, n_param_per_mol, idx_den, misc_names
        )
        self.prefix_molfit = prefix_molfit

    def call(self, params):
        mol_names, params_mol, params_dict = self.pm.derive_params(params)
        spectrum = self.call_check_params_dict(mol_names, params_mol, params_dict)
        if spectrum is not None:
            spectrum = spectrum[:, 1]
        return spectrum

    def call_full_output(self, params):
        mol_names, params_mol, params_dict = self.pm.derive_params(params)
        return self.call_params_dict(mol_names, params_mol, params_dict, return_full=True)

    def call_check_params_dict(self, mol_names, params_mol, params_dict):
        spectrum = self.call_params_dict(mol_names, params_mol, params_dict)
        if len(spectrum) != 0:
            return spectrum

        inds_include = []
        for idx in range(len(self.mol_names)):
            spectrum = self.call_params_dict(
                params_dict, params_mol[idx : idx+1], mol_names[idx: idx+1])
            if len(spectrum) != 0:
                inds_include.append(idx)
        if len(inds_include) == 0:
            return

        spectrum = self.call_params_dict(
            params_dict, params_mol[inds_include], mol_names[inds_include])
        return spectrum

    def call_params_dict(self, mol_names, params_mol, params_dict, return_full=False):
        fname_molfit = "{}_{}.molfit".format(self.prefix_molfit, os.getpid())
        create_molfit_file(fname_molfit, mol_names, params_mol)

        spectrum, log, trans, tau, job_dir = task_myXCLASS.myXCLASS(
            MolfitsFileName=fname_molfit, **params_dict, **self._xclass_kwargs
        )
        if return_full:
            return spectrum, log, trans, tau, job_dir
        else:
            shutil.rmtree(job_dir)
            return spectrum


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
                params_tmp[self.idx_den] *= params_iso[idx_iso]
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