from __future__ import annotations
import warnings
from typing import Optional, Callable
from pathlib import Path
from copy import deepcopy

import h5py
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from .embedding import create_embeding_model, EmbeddingV3
from .networks import create_parameter_estimator
from .. import cube
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
    results = inf_model.call_multi(
        inputs=inputs,
        postprocess=postprocess,
        pool=pool,
        device=device,
        disable_pbar=disable_pbar
    )
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
                obs_info, specie_list, sl_dict_list=[sl_dict]
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
                 postprocess: Callable,
                 loss_fn: str,
                 fname_cube: str,
                 species: str,
                 batch_size: int,
                 num_workers: int=2,
                 conn=None,
                 pool=None,
                 device=None):
    # Create data loader
    dataset = CubeDataset(
        fname=fname_cube,
        species=species,
        inf_model=inf_model,
        loss_fn=loss_fn,
    )
    # Ensure that one batch contains all species
    n_specie = len(species)
    batch_size = max(n_specie, batch_size)
    batch_size = batch_size//n_specie*n_specie
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_padding,
        shuffle=False,
        num_workers=num_workers,
    )
    return inf_model.call_multi(
        inputs=data_loader,
        postprocess=postprocess,
        n_specie=len(species),
        conn=conn,
        pool=pool,
        device=device,
    )


def collate_fn_padding(batch):
    embed_obs_batch = [torch.from_numpy(item[0]) for item in batch]
    embed_sl_batch = [torch.from_numpy(item[1]) for item in batch]
    embed_obs_batch = pad_sequence(embed_obs_batch, batch_first=True)
    embed_sl_batch = pad_sequence(embed_sl_batch, batch_first=True)
    mask = torch.all(embed_sl_batch == 0, dim=-1)
    others = tuple(zip(*[item[2:] for item in batch]))
    return (embed_obs_batch, embed_sl_batch, mask) + others


def _create_bound_info(config):
    param_names = ["theta", "T_ex", "N_tot", "delta_v", "v_LSR"]
    bound_info = {}
    for key, values in zip(param_names, config["nn"]["sampler"]["bounds"]):
        bound_info[key] = values
    return bound_info


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
            max_length=config["inference"].get("max_length", 4096)
        )
        embedding_model = create_embeding_model(
            ckpt["config"]["embedding"], sl_db=sl_db
        )
        warnings.warn("When the AI model is employed, the parameterization "
                      "and bound information in the config is overwritten.")
        config["sl_model"]["params"] = ckpt["config"]["sl_model"]["params"]
        config["bound_info"] = _create_bound_info(ckpt["config"])
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

    def call_multi(self,
                   inputs: list,
                   postprocess: Callable,
                   n_specie: int=1,
                   conn=None,
                   pool=None,
                   device: str=None,
                   disable_pbar: bool=False):
        def save_results(res):
            nonlocal idx_start
            if pool is not None:
                res = res.get()

            if conn is None:
                results.extend(res)
            else:
                conn.send(("save", (idx_start, res)))
                idx_start += len(res)

        n_wait = 3
        results = []
        wait_list = []
        idx_start = 0
        with tqdm(total=len(inputs), disable=disable_pbar) as pbar:
            pbar.set_description("Predicting")
            for embed_obs, embed_sl, mask, *args in inputs:
                if len(wait_list) == n_wait:
                    res = wait_list.pop(0)
                    save_results(res)
                    pbar.update()

                embed_obs = embed_obs.to(device)
                embed_sl = embed_sl.to(device)
                mask = mask.to(device)
                samps, log_prob, embed = self.draw_samples(
                    postprocess.n_draw, embed_obs, embed_sl, mask
                )

                if n_specie > 1:
                    # Reshape samps (B, N, D)
                    #   -> (B//n_specie, n_specie, N, D)
                    #   -> (B//n_specie, N, n_specie, D)
                    #   -> (B//n_specie, N, n_specie*D)
                    n_param = samps.shape[-1]
                    samps = samps.reshape(-1, n_specie, *samps.shape[1:])
                    samps = np.swapaxes(samps, 1, 2)
                    samps = samps.reshape(*samps.shape[:2], n_specie*n_param)
                    # Reshape log_prob (B, N) -> (B//n_specie, n_specie, N)
                    log_prob = log_prob.reshape(n_specie, -1)
                    log_prob = np.swapaxes(log_prob, 0, 1)
                    # Reshape embed (B, D) -> (B//n_specie, n_specie, D)
                    embed = embed.reshape(n_specie, -1)
                    embed = np.swapaxes(embed, 0, 1)
                    args = [x[::n_specie] for x in args]

                if pool is None:
                    wait_list.append(map(
                        lambda x: postprocess(*x),
                        zip(*args, samps, log_prob, embed)
                    ))
                else:
                    wait_list.append(pool.starmap_async(
                        postprocess,
                        zip(*args, samps, log_prob, embed)
                    ))
            while len(wait_list) > 0:
                res = wait_list.pop(0)
                save_results(res)
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


class CubeDataset(Dataset):
    def __init__(self,
                 fname: str,
                 species: list,
                 inf_model: InferenceModel,
                 loss_fn: str):
        self._fname = fname
        self._species = species
        self._loss_fn = loss_fn
        self._embedding_model = inf_model.embedding_model
        self._slm_factory = inf_model.slm_factory
        self._misc_data = cube.load_misc_data(fname)
        self._n_pixel = h5py.File(fname)["index"].shape[0]

    def __len__(self):
        return self._n_pixel*len(self._species)

    def __getitem__(self, idx):
        idx_pixel, idx_specie = divmod(idx, len(self._species))
        obs_info = cube.create_obs_info_from_cube(
            self._fname, idx_pixel, self._misc_data
        )
        embed_obs, embed_sl, sl_dict, specie_list \
            = self._embedding_model(obs_info, self._species[idx_specie])

        if len(self._species) == 1:
            fitting_model = self._slm_factory.create_fitting_model(
                obs_info, specie_list, self._loss_fn,
                sl_dict_list=[sl_dict]
            )
            return embed_obs, embed_sl, fitting_model

        specie_list = []
        for id_, name in enumerate(self._species):
            specie_list.append({"id": id_, "root": name, "species": [name]})
        fitting_model = self._slm_factory.create_fitting_model(
            obs_info, specie_list, self._loss_fn
        )
        return embed_obs, embed_sl, fitting_model

    @property
    def n_pixel(self):
        return self._n_pixel
