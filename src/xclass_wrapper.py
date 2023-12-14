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


def create_wrapper_from_config(spec_obs, mol_names, config, **kwargs):
    freq = spec_obs[:, 0].copy()
    freq_min = freq[0]
    freq_max = freq[-1]
    freq_step = freq[1] - freq[0]

    wrapper = XCLASSWrapper(
        FreqMin=freq_min,
        FreqMax=freq_max,
        FreqStep=freq_step,
        mol_names=mol_names,
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
                 mol_names, prefix_molfit,
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

        params_misc = []
        def set_misc_var(var, var_name):
            if var is None:
                params_misc.append(var_name)
            else:
                xclass_kwargs[var_name] = var

        set_misc_var(tBack, "tBack")
        set_misc_var(tslope, "tslope")
        set_misc_var(vLSR, "vLSR")
        self.params_misc = tuple(params_misc)

        self.mol_names = np.asarray(mol_names)
        self.n_param_per_mol = 5
        self.n_mol_param = len(mol_names)*self.n_param_per_mol
        self.prefix_molfit = prefix_molfit

    def call(self, params):
        params_dict, params_mol = self.derive_params_dict(params)
        spectrum = self.call_check_params_dict(params_dict, params_mol)
        if spectrum is not None:
            spectrum = spectrum[:, 1]
        return spectrum

    def call_full_output(self, params):
        params_dict, params_mol = self.derive_params_dict(params)
        return self.call_params_dict(params_dict, params_mol, self.mol_names, return_full=True)

    def derive_params_dict(self, params):
        params_mol = params[:self.n_mol_param]
        params_mol = params_mol.reshape(len(self.mol_names), -1)

        params_dict = {}
        for key, val in zip(self.params_misc, params[self.n_mol_param:]):
            params_dict[key] = val
        return params_dict, params_mol

    def call_check_params_dict(self, params_dict, params_mol):
        mol_names = self.mol_names
        spectrum = self.call_params_dict(params_dict, params_mol, mol_names)
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

    def call_params_dict(self, params_dict, params_mol, mol_names, return_full=False):
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