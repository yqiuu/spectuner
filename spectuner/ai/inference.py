from __future__ import annotations
from typing import Optional, Callable
from pathlib import Path
from copy import deepcopy

import h5py
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import spectuner_ml

from .embedding import create_embeding_model
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
        inputs.append(spectuner_ml.collate_fn_padding(batch))
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
                 save_spectra: bool=False,
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
        num_workers=2,
    )
    #
    results = inf_model.call_multi(data_loader, postprocess, pool, device)
    properties = np.zeros([len(results), 5])
    for idx, res in enumerate(results):
        properties[idx] = res["x"]

    return properties


def collate_fn_padding(batch):
    embed_obs_batch = [torch.from_numpy(item[0]) for item in batch]
    embed_sl_batch = [torch.from_numpy(item[1]) for item in batch]
    embed_obs_batch = pad_sequence(embed_obs_batch, batch_first=True)
    embed_sl_batch = pad_sequence(embed_sl_batch, batch_first=True)
    mask = torch.all(embed_sl_batch == 0, dim=-1)
    others = tuple(zip(*[item[2:] for item in batch]))
    return (embed_obs_batch, embed_sl_batch, mask) + others


def create_obs_info(freq_data, T_obs_data, T_bg_data, noise_data, beam_data):
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
        return self._n_pixel

    def __getitem__(self, idx):
        idx_specie, idx_pixel = divmod(idx, self.n_pixel)
        T_obs_data = []
        T_bg_data = []
        with h5py.File(self._fname) as fp:
            for i_segment in range(len(fp["cube"])):
                grp = fp["cube"][str(i_segment)]
                T_obs_data.append(np.array(grp["T_obs"][idx_pixel]))
                T_bg_data.append(np.array(grp["T_bg"][idx_pixel]))
        freq_data, noise_data, beam_data = self._misc_data
        obs_info = create_obs_info(
            freq_data=freq_data,
            T_obs_data=T_obs_data,
            T_bg_data=T_bg_data,
            noise_data=noise_data,
            beam_data=beam_data,
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
                 embedding_model: spectuner_ml.Embedding,
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
        model = spectuner_ml.create_parameter_estimator(
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
        return postprocess(fitting_model, samps[0], log_prob[0], embed[0])

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
    def embedding_model(self) -> spectuner_ml.Embedding:
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