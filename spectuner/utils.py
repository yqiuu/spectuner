import pickle


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