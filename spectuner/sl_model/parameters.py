from copy import deepcopy

import numpy as np

from .sl_model import derive_average_beam_size


class ParameterManager:
    """A class to decode parameters into a 2D array and apply scaling.

    Args:
        specie_list: Species information.
        param_info: A dict to indicate whether a parameter is shared and is
            log-scaled.
        obs_info: This is only used to derive the average beam size for the
            special parameterizations of theta. It can be ``None`` in other
            cases.
    """
    param_names = ["theta", "T_ex", "N_tot", "delta_v", "v_offset"]

    def __init__(self, specie_list: list,
                 param_info: dict,
                 obs_info: list | None):
        _specie_list = []
        assistants = []
        for item in specie_list:
            assistants.append(ParameterAssistant(item, param_info, obs_info))
            item = deepcopy(item)
            item["param_info"] = param_info
            _specie_list.append(item)
        self._specie_list = _specie_list
        self._assistants = assistants
        n_param = 0
        for asst in self._assistants:
            n_param += asst.n_param
        self._n_param = n_param

    @classmethod
    def from_config(cls, specie_list, config):
        return cls(specie_list, config["param_info"], config["obs_info"])

    @property
    def n_param(self) -> int:
        """Number of input parameters."""
        return self._n_param

    @property
    def specie_list(self) -> list:
        return self._specie_list

    def derive_params(self, params):
        """Decode input parameters into a 2D array and apply scaling."""
        assert len(params) == self.n_param, \
            f"Total number of parameters should be {self.n_param}."

        params_list = []
        idx_b = 0
        for asst in self._assistants:
            params_sub = asst.derive_params(params[idx_b : idx_b+asst.n_param])
            params_list.append(params_sub)
            idx_b += asst.n_param
        params_mol = np.vstack(params_list)
        return params_mol

    def recover_params(self, params_mol):
        params_ret = []
        idx_b = 0
        for item, asst in zip(self.specie_list, self._assistants):
            idx_e = idx_b + len(item["species"])
            params_ret.append(asst.recover_params(params_mol[idx_b:idx_e]))
            idx_b = idx_e
        params_ret = np.concatenate(params_ret)
        return params_ret

    def derive_bounds(self) -> np.ndarray:
        bounds_list = []
        for asst in self._assistants:
            bounds_list.extend(asst.derive_bounds())
        bounds = np.vstack(bounds_list)
        return bounds

    def get_subset_params(self, names, params):
        params_list = []
        idx_b = 0
        for asst in self._assistants:
            params_list.extend(asst.get_subset_params(
                names, params[idx_b : idx_b+asst.n_param]
            ))
            idx_b += asst.n_param
        if len(params_list) == 0:
            raise ValueError(f"Fail to find {names}.")
        return np.concatenate(params_list)


class ParameterAssistant:
    """A sub-class to decode parameters into a 2D array and apply scaling.

    Args:
        mol_item: Species information.
        param_info: A dict to indicate whether a parameter is shared and is
            log-scaled. If applicable, the code first loads ``param_info`` in
            ``mol_item`` and then use this argument as default.
        obs_info: This is only used to derive the average beam size for the
            special parameterizations of theta. It can be ``None`` in other
            cases.
    """
    def __init__(self, mol_item: dict,
                 param_info: dict,
                 obs_info: list | None):
        self.mol_item = mol_item
        param_names = ParameterManager.param_names
        #
        if "param_info" in mol_item:
            param_info = mol_item["param_info"]
        #
        inds_shared = []
        inds_private = []
        for idx, key in enumerate(param_names):
            if param_info[key]["is_shared"]:
                inds_shared.append(idx)
            else:
                inds_private.append(idx)
        self._inds_shared = inds_shared
        self._inds_private = inds_private
        self._n_shared_per_mol = len(inds_shared)
        self._n_private_per_mol = len(inds_private)
        self._n_param_per_mol = len(param_info)
        self._n_param = self._n_shared_per_mol \
            + len(mol_item["species"])*self._n_private_per_mol
        #
        self._scales = [param_info[key]["is_log"] for key in param_names]
        if "special" in param_info["theta"]:
            self._beam_size = derive_average_beam_size(obs_info)
            self._special = param_info["theta"]["special"]
            if self._special not in ["scaled", "eta"]:
                raise ValueError("Unknown name: {}.".format(self._special))
        else:
            self._beam_size = None
            self._special = None
        #
        self._bounds_dict = {key: param_info[key]["bound"] for key in param_names}

    @property
    def n_param(self) -> int:
        """Number of input parameters."""
        return self._n_param

    def derive_params(self, params):
        """Decode input parameters into a 2D array and apply scaling."""
        params_mol = self._derive_params(params)
        return self.forward_scaling(params_mol)

    def recover_params(self, params_mol):
        params_mol = self.reverse_scaling(params_mol)
        return self._recover_params(params_mol)

    def forward_scaling(self, params_mol):
        params_mol = params_mol.copy()
        for idx, is_log in enumerate(self._scales):
            if idx == 0 and self._special is not None:
                if self._special == "scaled":
                    if is_log:
                        params_mol[:, 0] = self._beam_size*10**params_mol[:, 0]
                    else:
                        params_mol[:, 0] = self._beam_size*params_mol[:, 0]
                elif self._special == "eta":
                    eta = params_mol[:, 0]
                    if is_log:
                        eta = 10**eta
                    params_mol[:, 0] = self._beam_size*np.sqrt(eta/(1 - eta))
                continue
            if is_log:
                params_mol[:, idx] = 10**params_mol[:, idx]
        return params_mol

    def reverse_scaling(self, params_mol):
        params_mol = params_mol.copy()
        for idx, is_log in enumerate(self._scales):
            if idx == 0 and self._special is not None:
                if self._special == "scaled":
                    if is_log:
                        params_mol[:, 0] = np.log10(params_mol[:, 0]/self._beam_size)
                    else:
                        params_mol[:, 0] = params_mol[:, 0]/self._beam_size
                elif self._special == "eta":
                    eta = np.square(params_mol[:, 0]/self._beam_size)
                    eta = eta/(1 + eta)
                    if is_log:
                        eta = np.log10(eta)
                    params_mol[:, 0] = eta
                continue
            if is_log:
                params_mol[:, idx] = np.log10(params_mol[:, idx])
        return params_mol

    def derive_bounds(self) -> list:
        param_names = ParameterManager.param_names
        bounds_list = []
        shared_names = [param_names[idx] for idx in self._inds_shared]
        private_names = [param_names[idx] for idx in self._inds_private]
        for name in shared_names:
            bounds_list.append(self._bounds_dict[name])
        for name in private_names:
            bounds_list.extend([self._bounds_dict[name]]*len(self.mol_item["species"]))
        return bounds_list

    def get_subset_params(self, names, params):
        params_list = []
        idx_mol = 0
        inds_mol = []
        for mol in self.mol_item["species"]:
            for name in names:
                if name == mol:
                    inds_mol.append(idx_mol)
            idx_mol += 1
        inds_mol = np.array(inds_mol)

        if len(inds_mol) > 0:
            n_mol = len(self.mol_item["species"])
            params_shared = params[:self._n_shared_per_mol]
            params_private = []
            offset = self._n_shared_per_mol
            for _ in range(self._n_private_per_mol):
                params_private.append(params[offset + inds_mol])
                offset += n_mol
            params_sub = np.append(params_shared, params_private)
            params_list.append(params_sub)
        return params_list

    def _derive_params(self, params):
        """Decode input parameters into a parameter list.

        For example, [123, AB, CD] > [[12AC3], [12BD3]].
        """
        n_mol = len(self.mol_item["species"])
        params_ret = np.zeros([n_mol, self._n_param_per_mol])
        #
        params_ret[:, self._inds_shared] = params[:self._n_shared_per_mol]
        #
        idx = self._n_shared_per_mol
        for idx_p in self._inds_private:
            params_ret[:, idx_p] = params[idx : idx+n_mol]
            idx += n_mol
        return params_ret

    def _recover_params(self, params):
        """Convert a parameter list to parameters.

        For example, [[12AC3], [12BD3]] > [123, AB, CD].
        """
        params_ret = [params[0, self._inds_shared]]
        for idx_p in self._inds_private:
            params_ret.append(params[:, idx_p])
        params_ret = np.concatenate(params_ret)
        return params_ret