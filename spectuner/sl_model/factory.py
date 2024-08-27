from functools import partial


from .sl_database import SpectralLineDatabase
from .sl_model import create_spectral_line_model_state, compute_effective_spectra


class SpectralLineModelFactory:
    def __init__(self, sl_db, freq_list, obs_info, trunc=10., eps_grid=1e-3):
        self._sl_database = SpectralLineDatabase(sl_db)
        self._freq_list = freq_list
        self._obs_info = obs_info
        self._trunc = trunc
        self._eps_grid = eps_grid

    def create(self, specie_list):
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
        return partial(compute_effective_spectra, slm_state)