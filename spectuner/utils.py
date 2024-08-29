import pickle

import h5py
import numpy as np


def load_pred_data(files, reset_id):
    if reset_id:
        files = list(files).copy()
        files.sort()
    pred_data_list = []
    for i_f, fname in enumerate(files):
        if is_exclusive(fname):
            continue
        pred_data = pickle.load(open(fname, "rb"))
        if reset_id:
            pred_data["specie"][0]["id"] = i_f
        pred_data_list.append(pred_data)
    return pred_data_list


def is_exclusive(fname):
    name = str(fname.name)
    return name.startswith("identify") \
        or name.startswith("combine") \
        or name.startswith("tmp")


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

        if isinstance(data, float) or isinstance(data, int):
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


def hdf_load_dict(fp, target_dict):
    for key, data in fp.items():
        if isinstance(data, h5py.Dataset):
            if data.shape == ():
                target_dict[key] = data[()]
            else:
                target_dict[key] = np.array(data)
        else:
            type_name = data.attrs["type"]
            if  type_name == "none":
                target_dict[key] = None
            elif type_name == "list":
                tmp = []
                for data_sub in data.values():
                    tmp.append(np.array(data_sub))
                target_dict[key] = tmp
            elif type_name == "dict":
                hdf_load_dict(data, target_dict)
            else:
                raise ValueError(f"Unknown type: {type_name}")