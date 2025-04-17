from __future__ import annotations
from typing import Callable
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import spectuner_ml

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
    results = inf_model(inputs, postprocess, pool, device, disable_pbar)
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


class InferenceModel:
    def __init__(self,
                 model: nn.Module,
                 embedding_model: spectuner_ml.Embedding,
                 slm_factory: SpectralLineModelFactory):
        self._model = model
        self._embedding_model = embedding_model
        self._slm_factory = slm_factory

    @classmethod
    def from_config(cls, config) -> InferenceModel:
        config_inf = config["inference"]
        ckpt = torch.load(
            config_inf["ckpt"],
            map_location="cpu",
            weights_only=True
        )
        model = spectuner_ml.create_parameter_estimator(ckpt["config"], is_sampler=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(config_inf["device"])
        model.eval()
        ckpt["config"]["embedding"].update(
            fname=config["inference"]["fname_db"],
            norms_sl=Path(__file__).parent/"normalizations_v1.yml",
            max_length=100000
        )
        embedding_model = spectuner_ml.create_embeding_model(ckpt["config"]["embedding"])
        config = deepcopy(config)
        config["sl_model"]["params"] = ckpt["config"]["sl_model"]["params"]
        slm_factory = SpectralLineModelFactory(config, sl_db=embedding_model.sl_db)
        return cls(model, embedding_model, slm_factory)

    def __call__(self, inputs, postprocess,
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