import numpy as np

from .sl_model import derive_average_beam_size


class ParameterManager:
    """A class to decode parameters into a 2D array and apply scaling."""
    param_names = ["theta", "T_ex", "N_tot", "delta_v", "v_LSR"]

    def __init__(self, specie_list, params_info, obs_info):
        self.specie_list = specie_list
        #
        inds_shared = []
        inds_private = []
        for idx, key in enumerate(self.param_names):
            if params_info[key]["is_shared"]:
                inds_shared.append(idx)
            else:
                inds_private.append(idx)
        self._inds_shared = inds_shared
        self._inds_private = inds_private
        self._n_shared_per_mol = len(inds_shared)
        self._n_private_per_mol = len(inds_private)
        self._n_param_per_mol = len(params_info)
        self._n_tot = self._derive_n_tot()
        #
        self._scales = [params_info[key]["is_log"] for key in self.param_names]
        if "special" in params_info["theta"]:
            self._beam_size = derive_average_beam_size(obs_info)
            self._special = params_info["theta"]["special"]
            if self._special not in ["scaled", "eta"]:
                raise ValueError("Unknown name: {}.".format(self._special))
        else:
            self._beam_size = None
            self._special = None

    @classmethod
    def from_config(cls, specie_list, config):
        return cls(specie_list, config["sl_model"]["params"], config["obs_info"])

    def derive_params(self, params):
        """Decode input parameters into a 2D array and apply scaling."""
        assert len(params) == self._n_tot, \
            f"Total number of parameters should be {self._n_tot}."

        params_list = []
        for item in self.specie_list:
            params_sub, idx = self._derive_params_sub(params, len(item["species"]))
            params_list.append(params_sub)
            params = params[idx:]
        params_mol = np.vstack(params_list)
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

    def derive_bounds(self, bounds_dict):
        bounds_list = []
        shared_names = [self.param_names[idx] for idx in self._inds_shared]
        private_names = [self.param_names[idx] for idx in self._inds_private]
        for item in self.specie_list:
            for name in shared_names:
                bounds_list.append(bounds_dict[name])
            for name in private_names:
                bounds_list.extend([bounds_dict[name]]*len(item["species"]))
        bounds = np.vstack(bounds_list)
        return bounds

    def get_subset_params(self, names, params):
        params_list = []
        idx_b = 0
        for item in self.specie_list:
            #
            idx_mol = 0
            inds_mol = []
            for mol in item["species"]:
                for name in names:
                    if name == mol:
                        inds_mol.append(idx_mol)
                idx_mol += 1
            inds_mol = np.array(inds_mol)
            #
            n_mol = len(item["species"])
            idx_e = idx_b + self._derive_n_param_sub(n_mol)
            params_sub = params[idx_b:idx_e]

            if len(inds_mol) > 0:
                params_shared = params_sub[:self._n_shared_per_mol]
                params_private = []
                offset = self._n_shared_per_mol
                for _ in range(self._n_private_per_mol):
                    params_private.append(params_sub[offset + inds_mol])
                    offset += n_mol
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

    def _derive_n_tot(self):
        n_tot = 0
        for item in self.specie_list:
            n_tot += self._derive_n_param_sub(len(item["species"]))
        return n_tot

    def _derive_n_param_sub(self, n_mol):
        return self._n_shared_per_mol + n_mol*self._n_private_per_mol