import numpy as np

from .sl_database import SpectralLineDatabase
from .sl_model import create_spectral_line_model_state, compute_effective_spectra


class SpectralLineModelFactory:
    """Spectral line model factory.

    Args:
        sl_db (str or SpectralLineDatabase): Path to the spectral line database.
            or a SpectralLineDatabase instance.
        freq_list (list): A list of arrays to specify the frequencies to compute
            the spectral line model.
        obs_info (list): A list of dicts to specify the observation information
            of each segment. The following keys should be included:
            - beam_info (float or tuple): Telescople size in meter or
            (BMAJ, BMIN) in degree.
            - T_bg (float): background temperature.
            - need_cmb (bool): Whethter to add additional CMB radiation in
            the continuum.

        params_info (dict): A dict to specify whether a parameter is
            shared and is in logarithmic scale. For example,
            {
                'theta': {'is_shared': False, 'is_log': True},
                'T_ex': {'is_shared': True, 'is_log': False},
                'N_tot': {'is_shared': True, 'is_log': True},
                'delta_v': {'is_shared': False, 'is_log': True},
                'v_LSR': {'is_shared': False, 'is_log': False}
            }
        trunc (flaot): Truncation of the Gaussian profile.
        eps_grid (float): Accuracy parameter for the adaptive quadrature to
            determine the grid points.
    """
    def __init__(self, sl_db, freq_list, obs_info, params_info, trunc=10., eps_grid=1e-3):
        if isinstance(sl_db, str):
            self._sl_database = SpectralLineDatabase(sl_db)
        elif isinstance(sl_db, SpectralLineDatabase):
            self._sl_database = sl_db
        else:
            raise ValueError(f"Invalid spectral line database: {sl_db}.")
        self._freq_list = freq_list
        self._obs_info = obs_info
        self._params_info = params_info
        self._trunc = trunc
        self._eps_grid = eps_grid

    def create(self, specie_list):
        """Create a ``SpectralLineModel`` instance.

        Args:
            specie_list (list): A list of dict that contains the following keys:
                - id (int): Number id.
                - root (str): Root entry name.
                - species (list): A list of entry names.

        Returns:
            SpectralLineModel: A ``SpectralLineModel`` instance.
        """
        sl_data_list = []
        for item in specie_list:
            for specie in item["species"]:
                sl_data_list.append(
                    self._sl_database.query(specie, self._freq_list)
                )
        slm_state = create_spectral_line_model_state(
            sl_data_list, self._freq_list, self._obs_info,
            self._trunc, self._eps_grid
        )
        param_mgr = ParameterManager(specie_list, self._params_info)
        return SpectralLineModel(slm_state, param_mgr)


class SpectralLineModel:
    def __init__(self, slm_state, param_mgr):
        self.slm_state = slm_state
        self.param_mgr = param_mgr

    def __call__(self, params):
        return compute_effective_spectra(
            self.slm_state, self.param_mgr.derive_params(params)
        )


class ParameterManager:
    """A class to decode parameters into a 2D array and apply scaling."""
    param_names = ["theta", "T_ex", "N_tot", "delta_v", "v_LSR"]

    def __init__(self, specie_list, params_info):
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