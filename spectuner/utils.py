import json

import h5py
import numpy as np


def load_result_list(fname, reset_id=False):
    """Load all fitting results exclusive of the combined results."""
    fp = h5py.File(fname)
    pred_data_list = []
    for i_f, (key, grp) in enumerate(fp.items()):
        if key.startswith("tmp") or key.startswith("combine"):
            continue
        pred_data = load_fitting_result(grp)
        if reset_id:
            pred_data["specie"][0]["id"] = i_f
        pred_data_list.append(pred_data)
    fp.close()
    return pred_data_list


def load_result_combine(fname):
    """Load the combined fitting result."""
    with h5py.File(fname) as fp:
        if "combine" in fp:
            pred_data = load_fitting_result(fp["combine"])
        else:
            pred_data = None
    return pred_data


def save_fitting_result(fp, res_dict):
    missing = []
    for key in ("specie", "freq", "T_pred", "x", "fun", "nfev"):
        if key not in res_dict:
            missing.append(key)
    if len(missing) > 0:
        raise ValueError(f"Missing keys: {missing}.")

    fp.create_dataset("specie", data=json.dumps(res_dict["specie"]))
    hdf_save_dict(fp, res_dict, ignore_list=["specie"])


def load_fitting_result(fp):
    ret_dict = {}
    ret_dict["specie"] = json.loads(fp["specie"][()])
    hdf_load_dict(fp, ret_dict, ignore_list=["specie"])
    return ret_dict


def hdf_save_dict(fp, save_dict, ignore_list=None):
    def save_array(fp, name, arr):
        if arr.dtype.kind == "f":
            fp.create_dataset(name, data=arr.astype("f4"))
        else:
            fp.create_dataset(name, data=arr)

    if ignore_list is None:
        ignore_list = []

    for key, data in save_dict.items():
        if key in ignore_list:
            continue

        key = f"{key}"
        if isinstance(data, float) or isinstance(data, int) \
            or isinstance(data, np.floating) or isinstance(data, np.integer):
            fp.create_dataset(key, data=data)
        elif isinstance(data, np.ndarray):
            save_array(fp, key, data)
        elif data is None:
            grp = fp.create_group(key)
            grp.attrs["type"] = "none"
        elif isinstance(data, list):
            grp = fp.create_group(key)
            grp.attrs["type"] = "list"
            for idx, item in enumerate(data):
                if not isinstance(item, np.ndarray):
                    raise ValueError(f"Unknown data: {item}.")
                save_array(grp, f"{idx}", item)
        elif isinstance(data, dict):
            grp = fp.create_group(key)
            grp.attrs["type"] = "dict"
            hdf_save_dict(grp, data, ignore_list)
        else:
            raise ValueError(f"Unknown data: '{key}': {data}.")


def hdf_load_dict(fp, load_dict, ignore_list=None):
    if ignore_list is None:
        ignore_list = []

    for key, data in fp.items():
        if key in ignore_list:
            continue

        if isinstance(data, h5py.Dataset):
            if data.shape == ():
                load_dict[key] = data[()]
            else:
                load_dict[key] = np.array(data)
        else:
            type_name = data.attrs["type"]
            if  type_name == "none":
                load_dict[key] = None
            elif type_name == "list":
                tmp = []
                for i_segment in range(len(data)):
                    tmp.append(np.array(data[f"{i_segment}"]))
                load_dict[key] = tmp
            elif type_name == "dict":
                load_dict_sub = {}
                load_dict[key] = load_dict_sub
                hdf_load_dict(data, load_dict_sub, ignore_list)
            else:
                raise ValueError(f"Unknown type: {type_name}")


def derive_specie_save_name(item):
    return "{}_{}".format(item["id"], item["root"])


def is_exclusive(fname):
    name = str(fname.name)
    return name.startswith("identify") \
        or name.startswith("combine") \
        or name.startswith("tmp")