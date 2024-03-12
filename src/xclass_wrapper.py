import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from copy import deepcopy

import numpy as np

# Import XCLASS
from xclass import task_myXCLASS, task_ListDatabase


def create_molfit_file(fname, mol_names, params, include_list, MaxElowSQL=None, MingASQL=None):
    text = ""
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


class XCLASSWrapper:
    def __init__(self, pm, prefix_molfit,
                 FreqMin=0., FreqMax=1., FreqStep=.1,
                 IsoTableFileName=None, **xclass_kwargs):
        self.pm = pm
        if IsoTableFileName is None:
            xclass_kwargs["iso_flag"] = False
        else:
            xclass_kwargs["iso_flag"] = True
        xclass_kwargs["IsoTableFileName"] = IsoTableFileName
        self.update_frequency(FreqMin, FreqMax, FreqStep)

        self._xclass_kwargs = xclass_kwargs
        self._MaxElowSQL = xclass_kwargs.pop("MaxElowSQL", None)
        self._MingASQL = xclass_kwargs.pop("MingASQL", None)
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
                yield None, None, None, None, None
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
            spec = None
        else:
            spec = spec[:, 1]
        return spec, log, trans, tau, job_dir

    def call_params_dict(self, mol_names, params_mol, params_dict, remove_dir):
        fname_molfit = "{}_{}.molfit".format(self.prefix_molfit, os.getpid())
        create_molfit_file(
            fname_molfit, mol_names, params_mol,
            self.include_list, self._MaxElowSQL, self._MingASQL
        )

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
    def __init__(self, mol_list, scaler, xclass_kwargs):
        n_param_per_mol = 5
        idx_den = 2
        misc_names = []
        for var_name in ["tBack", "tSlope", "vLSR"]:
            if not var_name in xclass_kwargs:
                misc_names.append(var_name)
        self._T_back = xclass_kwargs.get("tBack", 0.)

        # Set indices
        idx = 0
        n_mol_param = len(mol_list)*(n_param_per_mol - 1)
        self.inds_mol_param = slice(idx, n_mol_param)
        idx += n_mol_param

        n_den_param = 0
        for item in mol_list:
            n_den_param += len(item["molecules"])
        self.inds_den_param = slice(idx, idx + n_den_param)
        idx += n_den_param

        self.inds_misc_param = slice(idx, idx + len(misc_names))
        self.n_tot_param = idx + len(misc_names)

        # Set inds_mp
        inds_mp = list(range(n_param_per_mol))
        inds_mp.remove(idx_den)
        self.inds_mp = inds_mp

        # mol_names
        mol_names = []
        for item in mol_list:
            mol_names.extend(item["molecules"])
        self.mol_names = mol_names

        # Set id_list
        self.id_list = [item["id"] for item in mol_list]

        # Set den_inds
        den_inds = {}
        idx_b = 0
        for item in mol_list:
            mols = item["molecules"]
            if len(mols) == 0:
                continue

            idx_e = idx_b + len(mols)
            den_inds[item["id"]] = slice(idx_b, idx_e)
            idx_b = idx_e
        self.den_inds = den_inds

        #
        self.scaler = scaler
        self.mol_list = mol_list
        self.n_mol = len(mol_list)
        self.n_mol_param = n_mol_param
        self.n_den_param = n_den_param
        self.n_param_per_mol = n_param_per_mol

        self.idx_den = idx_den
        self.misc_names = misc_names
        self.n_misc_param = len(misc_names)

    @property
    def T_back(self):
        return self._T_back

    def derive_params(self, params):
        if len(params) != self.n_tot_param:
            raise ValueError(f"Total number of parameters should be {self.n_tot_param}.")

        params = self.scaler.derive_params(self, params)
        params_mol = self.derive_mol_params(params)
        params_misc = params[self.inds_misc_param]
        params_dict = {}
        for key, val in zip(self.misc_names, params_misc):
            params_dict[key] = val
        return self.mol_names, params_mol, params_dict

    def derive_mol_params(self, params):
        # params (N,)
        params_mol, params_den, _ = self.split_params(params, need_reshape=True)
        params_mol_ret = np.zeros([self.n_den_param, self.n_param_per_mol])
        idx = 0
        for idx_root, item in enumerate(self.mol_list):
            n_mol = len(item["molecules"])
            params_mol_ret[idx : idx+n_mol, self.inds_mp] = params_mol[idx_root]
            idx += n_mol
        params_mol_ret[:, self.idx_den] = params_den
        return params_mol_ret

    def split_params(self, params, need_reshape):
        params_mol = self.get_all_mol_params(params)
        if need_reshape:
            params_mol = params_mol.reshape(-1, self.n_param_per_mol - 1)
        params_den = self.get_all_den_params(params)
        params_misc = self.get_all_misc_params(params)
        return params_mol, params_den, params_misc

    def get_mol_slice(self, key):
        idx = self.id_list.index(key)
        n_param_per_mol = self.n_param_per_mol - 1
        return slice(idx*n_param_per_mol, (idx + 1)*n_param_per_mol)

    def get_den_slice(self, key):
        if key in self.den_inds:
            return self.den_inds[key]

    def get_all_mol_params(self, params):
        return params[self.inds_mol_param]

    def get_all_den_params(self, params):
        return params[self.inds_den_param]

    def get_all_misc_params(self, params):
        return params[self.inds_misc_param]

    def get_misc_params(self, key, params):
        params_misc = params[self.inds_misc_param]
        idx = self.misc_names.index(key)
        return params_misc[idx]

    def get_single_params(self, name, params):
        def find_item(name):
            idx = 0
            for item in self.mol_list:
                for mol in item["molecules"]:
                    if name == mol:
                        return item, idx
                    idx += 1
        item, idx = find_item(name)
        params_mol = params[self.get_mol_slice(item["id"])]
        params_den = self.get_all_den_params(params)[idx]
        params_single = np.append(params_mol, [params_den])
        return params_single


class Scaler:
    def derive_params(self, pm, params):
        params_mol, params_den, params_misc = pm.split_params(params, need_reshape=True)
        params_mol = params_mol.copy()
        params_mol[:, :3] = 10**params_mol[:, :3]
        params_den = 10**params_den
        params_new = np.concatenate([np.ravel(params_mol), params_den, params_misc])
        return params_new

    def derive_bounds(self, pm, bounds_mol, bounds_den, bounds_misc):
        # bounds_mol (5, 2)
        # bounds_den (2,)
        # bounds_misc (dict)
        bounds_mol = np.tile(bounds_mol, (pm.n_mol, 1))
        bounds_den = np.tile(bounds_den, (pm.n_den_param, 1))

        bounds_misc_ = []
        for key in pm.misc_names:
            bounds_misc_.append(bounds_misc[key])
        if len(bounds_misc_) == 0:
            bounds_misc = np.zeros((0, 2))
        else:
            bounds_misc = np.vstack(bounds_misc_)

        bounds = np.vstack([bounds_mol, bounds_den, bounds_misc])
        return bounds


@dataclass(frozen=True)
class MoleculeStore:
    mol_list: list
    include_list: list
    scaler: object

    def create_parameter_manager(self, xclass_kwargs):
        return ParameterManager(self.mol_list, self.scaler, xclass_kwargs)

    def create_spectral_line_model(self, xclass_kwargs):
        pm = self.create_parameter_manager(xclass_kwargs)
        sl_model = XCLASSWrapper(pm, **xclass_kwargs)
        return sl_model

    def select_single(self, name):
        def find_item(name):
            for item in self.mol_list:
                for mol in item["molecules"]:
                    if name == mol:
                        return item
        item = deepcopy(find_item(name))
        item["root"] = name
        item["molecules"] = [name]
        mol_list_new = [item]
        #
        include_list_new = []
        for in_list in self.include_list:
            if name in in_list:
                include_list_new.append([name])
            else:
                include_list_new.append([])

        return MoleculeStore(mol_list_new, include_list_new, self.scaler)

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

        return MoleculeStore(mol_list_new, include_list_new, self.scaler)
