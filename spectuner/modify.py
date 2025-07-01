from pathlib import Path

import h5py
import numpy as np

from .utils import load_result_combine, save_fitting_result, load_result_list
from .slm_factory import (
    combine_specie_lists, derive_sub_specie_list_with_params,
    SpectralLineModelFactory
)
from .identify import identify


__all__ = ["modify"]


def modify(config, result_dir, sl_db=None):
    """Modify a combined result.

    Args:
        config (dict): Config.
        result_dir (str): Directory of all fitting result.
    """
    config_modify = config["modify"]
    exclude_id_list = config_modify["exclude_id_list"]
    exclude_name_set = set(config_modify["exclude_name_list"])

    result_dir = Path(result_dir)
    fname = result_dir/"results_combine.h5"
    data_combine = load_result_combine(fname)
    freq_data = data_combine["freq"]
    specie_list = data_combine["specie"]

    names_in = []
    for item in specie_list:
        species = []
        if item["id"] not in exclude_id_list:
            species.extend(item["species"])
        species = [name for name in species if name not in exclude_name_set]
        names_in.extend(species)

    slm_factory = SpectralLineModelFactory(config, sl_db=sl_db)
    obs_info = config["obs_info"]
    specie_list_new, params_new = derive_sub_specie_list_with_params(
        slm_factory, obs_info, specie_list, names_in, data_combine["x"]
    )

    include_id_list = config_modify["include_id_list"]
    if len(include_id_list) != 0:
        specie_lists = [specie_list_new]
        params_list = [params_new]
        data_list = []
        for pred_data in load_result_list(fname):
            if pred_data["specie"][0]["id"] in include_id_list:
                data_list.append(pred_data)
        for data in data_list:
            specie_list = data["specie"]
            params = data["x"]
            names_in = set(specie_list[0]["species"]) - exclude_name_set
            specie_list, params = derive_sub_specie_list_with_params(
                slm_factory, obs_info, specie_list, names_in, params
            )
            specie_lists.append(specie_list)
            params_list.append(params)
        specie_list_new, params_new \
            = combine_specie_lists(specie_lists, params_list)

    sl_model = slm_factory.create_sl_model(obs_info, specie_list_new)
    T_pred_data = sl_model(params_new)
    save_dict = {
        "specie": specie_list_new,
        "freq": freq_data,
        "T_pred": T_pred_data,
        "x": params_new,
        "fun": np.nan,
        "nfev": 0,
    }
    save_name = result_dir/Path("combine_modified.h5")
    with h5py.File(save_name, "w") as fp:
        save_fitting_result(fp.create_group("combine"), save_dict)

    identify(config, save_name, mode="combine")