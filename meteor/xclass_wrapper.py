import os
import warnings
import shutil
from collections import defaultdict
from dataclasses import dataclass
from copy import deepcopy

import numpy as np

# Import XCLASS
try:
    from xclass import task_myXCLASS
except ImportError:
    warnings.warn("XCLASS is not installed.")


def create_molfit_file(fname, mol_names, params, include_list,
                       MinNumTransitionsSQL=None, MaxNumTransitionsSQL=None,
                       TransOrderSQL=None, MaxElowSQL=None, MingASQL=None):
    text = ""
    if MinNumTransitionsSQL is not None:
        text += f"%%MinNumTransitionsSQL = {MinNumTransitionsSQL}\n"
    if MaxNumTransitionsSQL is not None:
        text += f"%%MaxNumTransitionsSQL = {MaxNumTransitionsSQL}\n"
    if TransOrderSQL is not None:
        text += f"%%TransOrderSQL = {TransOrderSQL}\n"
    if MaxElowSQL is not None:
        text += f"%%MaxElowSQL = {MaxElowSQL}\n"
    if MingASQL is not None:
        text += f"%%MingASQL = {MingASQL}\n"
    text += f"""% name of molecule		number of components
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

    n_param_per_mol = 5
    idx_den = 2
    pm = ParameterManager(
        mol_dict, n_param_per_mol, idx_den, config
    )
    wrapper = XCLASSWrapper(
        pm=pm,
        FreqMin=freq_min,
        FreqMax=freq_max,
        FreqStep=freq_step,
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


def combine_mol_stores(mol_store_list, params_list):
    mol_list = []
    include_list = [[] for _ in range(len(mol_store_list[0].include_list))]
    for mol_store in mol_store_list:
        mol_list.extend(mol_store.mol_list)
        for in_list_new, in_list in zip(include_list, mol_store.include_list):
            in_list_new.extend(in_list)
    mol_store_new = MoleculeStore(mol_list, include_list)
    params_new = np.concatenate(params_list)
    return mol_store_new, params_new


class XCLASSWrapper:
    def __init__(self, pm, prefix_molfit,
                 FreqMin=0., FreqMax=1., FreqStep=.1,
                 IsoTableFileName=None, **kwargs):
        self.pm = pm
        kwargs_ = {
            "NoSubBeamFlag": True,
            "printFlag": False,
        }
        if IsoTableFileName is None:
            kwargs["iso_flag"] = False
        else:
            kwargs["iso_flag"] = True
        kwargs["IsoTableFileName"] = IsoTableFileName
        kwargs_.update(**kwargs)

        self.update_frequency(FreqMin, FreqMax, FreqStep)

        kwargs_molfit = {}
        names = [
            "MinNumTransitionsSQL",
            "MaxNumTransitionsSQL",
            "TransOrderSQL",
            "MaxElowSQL",
            "MingASQL",
        ]
        for key in names:
            kwargs_molfit[key] = kwargs_.pop(key, None)
        self._kwargs_molfit = kwargs_molfit

        self._kwargs_xclass = kwargs_
        self.prefix_molfit = prefix_molfit
        self.include_list = None

    def update_frequency(self, freq_min, freq_max, freq_step):
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.freq_step = freq_step

    def update_include_list(self, include_list):
        self.include_list = include_list

    def call_multi(self, freq_data, include_list, params, remove_dir=True):
        for freq, in_list in zip(freq_data, include_list):
            if len(in_list) == 0:
                spec = self.get_default_spec()
                yield spec, None, None, None, None
            else:
                self.update_frequency(*derive_freq_range(freq))
                self.update_include_list(in_list)
                yield self.call(params, remove_dir)

    def call(self, params, remove_dir=True):
        mol_names, params_mol, params_dict = self.pm.derive_params(params)
        spec, log, trans, tau, job_dir = self.call_params_dict(
            mol_names, params_mol, params_dict, remove_dir
        )
        if len(spec) == 0:
            spec = self.get_default_spec()
        else:
            spec = spec[:, 1]
        return spec, log, trans, tau, job_dir

    def call_params_dict(self, mol_names, params_mol, params_dict, remove_dir):
        fname_molfit = "{}_{}.molfit".format(self.prefix_molfit, os.getpid())
        create_molfit_file(
            fname_molfit, mol_names, params_mol,
            self.include_list, **self._kwargs_molfit
        )

        spec, log, trans, tau, job_dir = task_myXCLASS.myXCLASSCore(
            FreqMin=self.freq_min,
            FreqMax=self.freq_max,
            FreqStep=self.freq_step,
            MolfitsFileName=fname_molfit,
            **params_dict,
            **self._kwargs_xclass
        )
        if remove_dir:
            shutil.rmtree(job_dir)
        return spec, log, trans, tau, job_dir

    def create_molfit_file(self, fname, params):
        mol_names, params_mol, _ = self.pm.derive_params(params)
        create_molfit_file(fname, mol_names, params_mol, self.include_list)

    def get_default_spec(self):
        freq = np.arange(self.freq_min, self.freq_max, self.freq_step)
        return np.full_like(freq, self.pm.T_back)


class ParameterManager:
    param_names = ["theta", "T_ex", "N_tot", "delta_v", "v_LSR"]

    def __init__(self, mol_list, config_params, config_slm):
        self.mol_list = mol_list
        self.mol_names = self._derive_mol_names()
        #
        self._T_back = config_slm.get("tBack", 0.)
        #
        inds_shared = []
        inds_private = []
        for idx, key in enumerate(self.param_names):
            if config_params[key]["is_shared"]:
                inds_shared.append(idx)
            else:
                inds_private.append(idx)
        self._inds_shared = inds_shared
        self._inds_private = inds_private
        self._n_shared_per_mol = len(inds_shared)
        self._n_private_per_mol = len(inds_private)
        self._n_param_per_mol = len(config_params)
        self._n_tot = self._derive_n_tot()
        #
        self._scales = [config_params[key]["is_log"] for key in self.param_names]

    @property
    def T_back(self):
        return self._T_back

    def derive_params(self, params):
        """Decode input parameters and apply scaling.

        This is the primary method to interact with xclass.
        """
        params_mol = self.derive_mol_params(params)
        return self.mol_names, params_mol, {}

    def derive_mol_params(self, params):
        """Decode input parameters into a parameter list."""
        assert len(params) == self._n_tot, \
            f"Total number of parameters should be {self._n_tot}."

        params_list = []
        for item in self.mol_list:
            params_sub, idx = self._derive_params_sub(params, len(item["molecules"]))
            params_list.append(params_sub)
            params = params[idx:]
        params_mol = np.vstack(params_list)
        for idx, is_log in enumerate(self._scales):
            if is_log:
                params_mol[:, idx] = 10**params_mol[:, idx]
        return params_mol

    def derive_bounds(self, bounds_dict):
        bounds_list = []
        shared_names = [self.param_names[idx] for idx in self._inds_shared]
        private_names = [self.param_names[idx] for idx in self._inds_private]
        for item in self.mol_list:
            for name in shared_names:
                bounds_list.append(bounds_dict[name])
            for name in private_names:
                bounds_list.extend([bounds_dict[name]]*len(item["molecules"]))
        bounds = np.vstack(bounds_list)
        return bounds

    def get_subset_params(self, names, params):
        params_list = []
        idx_b = 0
        for item in self.mol_list:
            #
            idx_mol = 0
            inds_mol = []
            for mol in item["molecules"]:
                for name in names:
                    if name == mol:
                        inds_mol.append(idx_mol)
                idx_mol += 1
            #
            n_mol = len(item["molecules"])
            idx_e = idx_b + self._derive_n_param_sub(n_mol)
            params_sub = params[idx_b:idx_e]

            if len(inds_mol) > 0:
                params_shared = params_sub[:self._n_shared_per_mol]
                params_private = []
                for idx_mol in inds_mol:
                    idx_p = self._n_shared_per_mol + idx_mol
                    for _ in range(self._n_private_per_mol):
                        params_private.append(params_sub[idx_p])
                        idx_p += n_mol
                params_sub = np.append(params_shared, params_private)
                params_list.append(params_sub)
            #
            idx_b = idx_e
        if len(params_list) == 0:
            raise ValueError(f"Fail to find {name}.")
        return np.concatenate(params_list)

    def _derive_params_sub(self, params, n_mol):
        """Decode input parameters into a parameter list.

        For example, [123, AB, CD] > [[12AB3], [12CD3]].
        """
        params_ret = np.zeros([n_mol, self._n_param_per_mol])
        #
        params_ret[:, self._inds_shared] = params[:self._n_shared_per_mol]
        #
        idx = self._n_shared_per_mol
        for idx_p in self._inds_private:
            params_ret[:, idx_p] = params[idx : idx+n_mol]
            idx += n_mol
        return params_ret, idx

    def _derive_mol_names(self):
        mol_names = []
        for item in self.mol_list:
            mol_names.extend(item["molecules"])
        return mol_names

    def _derive_n_tot(self):
        n_tot = 0
        for item in self.mol_list:
            n_tot += self._derive_n_param_sub(len(item["molecules"]))
        return n_tot

    def _derive_n_param_sub(self, n_mol):
        return self._n_shared_per_mol + n_mol*self._n_private_per_mol


@dataclass(frozen=True)
class MoleculeStore:
    mol_list: list
    include_list: list

    def create_parameter_manager(self, config):
        return ParameterManager(self.mol_list, config["params"], config["sl_model"])

    def create_spectral_line_model(self, config):
        param_mgr = self.create_parameter_manager(config)
        sl_model = XCLASSWrapper(param_mgr, **config["sl_model"])
        return sl_model

    def compute_T_pred_data(self, params, freq_data, config):
        sl_model = self.create_spectral_line_model(config)
        iterator = sl_model.call_multi(
            freq_data, self.include_list, params, remove_dir=True
        )
        return [args[0] for args in iterator]

    def select_subset(self, names):
        mol_list_new = []
        for item in self.mol_list:
            mols_new = [mol for mol in item["molecules"] if mol in names]
            if len(mols_new) > 0:
                item_new = deepcopy(item)
                item_new["molecules"] = mols_new
                mol_list_new.append(item_new)
        #
        include_list_new = []
        for in_list in self.include_list:
            include_list_new.append([mol for mol in in_list if mol in names])

        return MoleculeStore(mol_list_new, include_list_new)

    def select_subset_with_params(self, names, params, config):
        mol_store_sub = self.select_subset(names)
        param_mgr = self.create_parameter_manager(config)
        params_sub = param_mgr.get_subset_params(names, params)
        return mol_store_sub, params_sub