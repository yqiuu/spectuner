from __future__ import annotations
from typing import Optional, Callable
from pathlib import Path
from copy import copy, deepcopy

import h5py
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .embedding import create_embeding_model, EmbeddingV3
from .networks import create_parameter_estimator
from ..preprocess import preprocess_spectrum
from ..sl_model import SpectralLineDB
from ..slm_factory import SpectralLineModelFactory


def predict_single_pixel(inf_model: InferenceModel,
                         obs_info: list,
                         entries: list,
                         numbers: list,
                         max_diff: int,
                         max_batch_size: int,
                         postprocess: Callable,
                         pool=None,
                         device=None,
                         disable_pbar=False):
    inds_sorted = np.argsort(numbers)[::-1]
    lookup_re = {idx: i_re for i_re, idx in enumerate(inds_sorted)}

    entries_sorted = np.asarray(entries)[inds_sorted]
    numbers_sorted = np.asarray(numbers)[inds_sorted]
    inputs = prepare_list_by_number(
        inf_model=inf_model,
        obs_info=obs_info,
        entries_sorted=entries_sorted,
        numbers_sorted=numbers_sorted,
        max_diff=max_diff,
        max_batch_size=max_batch_size
    )
    results = inf_model.call_multi(inputs, postprocess, pool, device, disable_pbar)
    results = [results[lookup_re[idx]] for idx in range(len(results))]
    return results


def prepare_list_by_number(inf_model: InferenceModel,
                           obs_info: list,
                           entries_sorted: np.ndarray,
                           numbers_sorted: np.ndarray,
                           max_diff: int,
                           max_batch_size: int):
    entry_list = split_list_by_number(
        entries_sorted, numbers_sorted, max_diff, max_batch_size
    )
    inputs = []
    for sub_list in entry_list:
        batch = []
        for name in sub_list:
            embed_obs, embed_sl, sl_dict, specie_list \
                = inf_model.embedding_model(obs_info, name)
            fitting_model = inf_model.slm_factory.create_fitting_model(
                obs_info, specie_list, [sl_dict]
            )
            batch.append((embed_obs, embed_sl, fitting_model))
        inputs.append(collate_fn_padding(batch))
    return inputs


def split_list_by_number(lst1, lst2, max_diff, max_batch_size):
    """Splits the first list into sublists based on the second list's values.

    The second list is assumed to be in descending order. Each sublist is formed
    such that:
    1. The difference between the maximum and minimum values in the
    corresponding sublist of the second list is less than `max_diff`.
    2. The length of the sublist is less than `max_batch_size`.

    Args:
        lst1 (list): The first list, which can contain elements of any type.
        lst2 (list): The second list, a list of integers in descending order.
        max_diff (int): The maximum allowed difference between the maximum and minimum
                       values in the sublist of `lst2`.
        max_batch_size (int): The maximum allowed length of the sublist.

    Returns:
        list: A list of sublists, where each sublist satisfies the criteria.
    """
    result = []
    start = 0

    while start < len(lst1):
        end = start
        while end < len(lst1):
            current_diff = lst2[start] - lst2[end]
            if current_diff >= max_diff or end + 1 - start > max_batch_size:
                break
            end += 1

        result.append(lst1[start:end])
        start = end

    return result


def predict_cube(inf_model: InferenceModel,
                 fname_cube: str,
                 species: str,
                 batch_size: int,
                 postprocess: Callable,
                 num_workers: int=2,
                 need_spectra: bool=True,
                 pool=None,
                 device=None):
    # Create data loader
    dataset = CubeDataset(
        fname=fname_cube,
        species=species,
        inf_model=inf_model,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_padding,
        shuffle=False,
        num_workers=num_workers,
    )

    #
    postprocess_ = _AddExtraProps(
        inf_model.slm_factory, postprocess, need_spectra
    )
    results = inf_model.call_multi(data_loader, postprocess_, pool, device)

    # Format results
    res_dict = {}
    for res in results:
        name = res["specie"][0]["root"]
        if name not in res_dict:
            res_dict[name] = {
                "params": [],
                "t1_score": [],
                "t2_score": [],
                "t3_score": [],
                "t4_score": [],
                "s_tp_tot": [],
                "num_tp": [],
                "s_fp_tot": [],
                "num_fp": []
            }
            if need_spectra:
                res_dict[name]["T_pred"] = {
                    f"{idx}": [] for idx in range(len(res["T_pred"]))
                }

        sub_dict = res_dict[name]
        for key in sub_dict:
            if key != "T_pred":
                sub_dict[key].append(res[key])
            elif need_spectra:
                for idx, T_pred in enumerate(res["T_pred"]):
                    sub_dict["T_pred"][f"{idx}"].append(T_pred)

    for sub_dict in res_dict.values():
        for key, val in sub_dict.items():
            if key == "params":
                sub_dict[key] = np.vstack(val)
            elif key == "T_pred":
                for idx, T_data in sub_dict[key].items():
                    sub_dict[key][idx] = np.vstack(T_data)
            else:
                sub_dict[key] = np.asarray(val)
    return res_dict


def collate_fn_padding(batch):
    embed_obs_batch = [torch.from_numpy(item[0]) for item in batch]
    embed_sl_batch = [torch.from_numpy(item[1]) for item in batch]
    embed_obs_batch = pad_sequence(embed_obs_batch, batch_first=True)
    embed_sl_batch = pad_sequence(embed_sl_batch, batch_first=True)
    mask = torch.all(embed_sl_batch == 0, dim=-1)
    others = tuple(zip(*[item[2:] for item in batch]))
    return (embed_obs_batch, embed_sl_batch, mask) + others


def create_obs_info_from_cube(fname: str, idx_pixel:int , misc_data: list):
        T_obs_data = []
        T_bg_data = []
        with h5py.File(fname) as fp:
            for i_segment in range(len(fp["cube"])):
                grp = fp["cube"][str(i_segment)]
                T_obs_data.append(np.array(grp["T_obs"][idx_pixel]))
                T_bg_data.append(np.array(grp["T_bg"][idx_pixel]))
        freq_data, noise_data, beam_data = misc_data

        obs_info = []
        for i_segment, freq in enumerate(freq_data):
            spec = np.vstack([freq, T_obs_data[i_segment]]).T
            spec = preprocess_spectrum(spec)
            obs_info.append({
                "spec": spec,
                "beam_info": beam_data[i_segment],
                "T_bg": T_bg_data[i_segment],
                "need_cmb": True,
                "noise": noise_data[i_segment],
            })
        return obs_info


def load_misc_data(fname):
    """
    Returns:
        freq_data (list):
        noise_data (list):
        beam_data (list):
    """
    with h5py.File(fname) as fp:
        freq_data = []
        noise_data = []
        beam_data = []
        for i_segment in range(len(fp["cube"])):
            grp = fp["cube"][str(i_segment)]
            freq_data.append(np.array(grp["freq"]))
            noise_data.append(np.array(grp["noise"]))
            beam_data.append(np.array(grp["beam"]))
    return freq_data, noise_data, beam_data


class CubeDataset(Dataset):
    def __init__(self, fname: str, species: str, inf_model: InferenceModel):
        self._fname = fname
        self._species = species
        self._embedding_model = inf_model.embedding_model
        self._slm_factory = inf_model.slm_factory
        self._misc_data = load_misc_data(fname)
        self._n_pixel = h5py.File(fname)["index"].shape[0]

    def __len__(self):
        return self._n_pixel*len(self._species)

    def __getitem__(self, idx):
        idx_specie, idx_pixel = divmod(idx, self.n_pixel)
        obs_info = create_obs_info_from_cube(
            self._fname, idx_pixel, self._misc_data
        )
        embed_obs, embed_sl, sl_dict, specie_list \
            = self._embedding_model(obs_info, self._species[idx_specie])
        fitting_model = self._slm_factory.create_fitting_model(
            obs_info, specie_list, [sl_dict]
        )
        return embed_obs, embed_sl, fitting_model

    @property
    def n_pixel(self):
        return self._n_pixel


class InferenceModel:
    def __init__(self,
                 model: nn.Module,
                 embedding_model: EmbeddingV3,
                 slm_factory: SpectralLineModelFactory):
        self._model = model
        self._embedding_model = embedding_model
        self._slm_factory = slm_factory

    @classmethod
    def from_config(cls,
                    config: dict,
                    sl_db: Optional[SpectralLineDB]=None) -> InferenceModel:
        config_inf = config["inference"]
        ckpt = torch.load(
            config_inf["ckpt"],
            map_location="cpu",
            weights_only=True
        )
        model = create_parameter_estimator(
            ckpt["config"], is_sampler=False
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(config_inf["device"])
        model.eval()
        ckpt["config"]["embedding"].update(
            fname=config["sl_model"]["fname_db"],
            norms_sl=Path(__file__).parent/"normalizations_v1.yml",
            max_length=100000
        )
        embedding_model = create_embeding_model(
            ckpt["config"]["embedding"], sl_db=sl_db
        )
        config = deepcopy(config)
        config["sl_model"]["params"] = ckpt["config"]["sl_model"]["params"]
        slm_factory = SpectralLineModelFactory(config, sl_db=embedding_model.sl_db)
        return cls(model, embedding_model, slm_factory)

    def call_single(self,
                    obs_info: list,
                    specie_name: str,
                    postprocess: Callable,
                    T_base_data: Optional[list]=None,
                    device: Optional[str]=None) -> dict:
        if T_base_data is None:
            obs_info_ = obs_info
        else:
            obs_info_ = deepcopy(obs_info)
            for item, T_base in zip(obs_info_, T_base_data):
                item["spec"][:, 1] -= T_base

        embed_obs, embed_sl, sl_dict, specie_list \
                = self.embedding_model(obs_info_, specie_name)
        fitting_model = self.slm_factory.create_fitting_model(
            obs_info, specie_list, [sl_dict], T_base_data=T_base_data
        )
        embed_obs = torch.from_numpy(embed_obs).unsqueeze(0).to(device)
        embed_sl = torch.from_numpy(embed_sl).unsqueeze(0).to(device)
        mask = None
        samps, log_prob, embed = self.draw_samples(
            postprocess.n_draw, embed_obs, embed_sl, mask
        )
        res_dict = postprocess(fitting_model, samps[0], log_prob[0], embed[0])
        res_dict["T_base"] = T_base_data
        return res_dict

    def call_multi(self, inputs, postprocess,
                   pool=None, device=None, disable_pbar=False):
        results = []
        wait_list = []
        total = len(inputs)
        if pool is not None:
            total *= 2
        with tqdm(total=total, disable=disable_pbar) as pbar:
            for embed_obs, embed_sl, mask, *args in inputs:
                pbar.set_description(
                    "Predicting (batch_size={})".format(len(embed_obs))
                )
                embed_obs = embed_obs.to(device)
                embed_sl = embed_sl.to(device)
                mask = mask.to(device)
                samps, log_prob, embed = self.draw_samples(
                    postprocess.n_draw, embed_obs, embed_sl, mask
                )
                if pool is None:
                    results.extend(map(
                        lambda x: postprocess(*x),
                        zip(*args, samps, log_prob, embed))
                    )
                else:
                    wait_list.append(pool.starmap_async(
                        postprocess,
                        zip(*args, samps, log_prob, embed)
                    ))
                pbar.update()

            if pool is not None:
                pbar.set_description("Optimizing")
                for res in wait_list:
                    results.extend(res.get())
                    pbar.update()

        return results

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def embedding_model(self) -> EmbeddingV3:
        return self._embedding_model

    @property
    def slm_factory(self) -> SpectralLineModelFactory:
        return self._slm_factory

    @property
    def postprocess(self) -> Callable:
        return self._postprocess

    def draw_samples(self, n_draw, embed_obs, embed_sl, mask):
        with torch.no_grad():
            embed = self.model(
                embed_obs=embed_obs,
                embed_sl=embed_sl,
                key_padding_mask=mask
            )
            samps, log_prob = self.model(
                n_draw=n_draw, embed=embed
            )
        embed = embed.cpu().numpy()
        samps = samps.cpu().numpy()
        log_prob = log_prob.cpu().numpy()
        return samps, log_prob, embed


class _AddExtraProps:
    def __init__(self,
                 slm_factory: SpectralLineModelFactory,
                 postprocess: Callable,
                 need_spectra: bool):
        self._slm_factory = copy(slm_factory)
        self._slm_factory._sl_db = None # Prevent copying the database from multiprocessing
        self._postprocess = postprocess
        self._need_spectra = need_spectra

    def __call__(self, fitting_model, *args):
        res = self._postprocess(fitting_model, *args)
        params = fitting_model.sl_model.param_mgr.derive_params(res["x"])[0]
        # Set column density to log10, ensure that the index is correct
        params[2] = np.log10(params[2])
        res["params"] = params
        peak_mgr = self._slm_factory.create_peak_mgr(fitting_model.obs_info)
        scores_tp, scores_fp = peak_mgr.compute_score_all(
            res["T_pred"], use_f_dice=True
        )
        scores_tp = np.sort(scores_tp)[::-1]
        n_max = 4
        for idx in range(n_max):
            res[f"t{idx+1}_score"] \
                = scores_tp[idx] if len(scores_tp) > idx + 1 else 0.
        res["s_tp_tot"] = np.sum(scores_tp)
        res["num_tp"] = len(scores_tp)
        res["s_fp_tot"] = np.sum(scores_fp)
        res["num_fp"] = len(scores_fp)
        if not self._need_spectra:
            del res["T_pred"]
        return res

    @property
    def n_draw(self):
        return self._postprocess.n_draw