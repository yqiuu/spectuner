from copy import deepcopy

import torch
from torch import nn

from .basics import SwiGLU
from .backbone_v6 import BackboneV6
from .flows import create_transformer_flow


def create_parameter_estimator(config, is_sampler):
    config_backbone = config["nn"]["backbone"]
    if "d_obs" in config_backbone:
        config_backbone["d_obs"] = config["embedding"]["n_grid"]
    config_backbone = deepcopy(config_backbone)
    name = config_backbone.pop("name", "v0")
    if name == "v6":
        backbone = BackboneV6(**config_backbone)
    else:
        raise ValueError(f"Unknown backbone model: {name}.")
    config_sampler = config["nn"]["sampler"]
    sampler = create_sampler(config_sampler, config_backbone, is_sampler)
    if config["nn"].get("need_vf", False):
        value_func = ValueFunction(
            d_model=config_backbone["d_model"]
        )
    else:
        value_func = None

    return ParameterEstimator(backbone, sampler, value_func)


def create_sampler(config_sampler, config_backbone, is_samper):
    d_model = config_backbone["d_model"]

    kwargs_flow = deepcopy(config_sampler)
    if "width_min" not in kwargs_flow:
        kwargs_flow["width_min"] = 0.
    kwargs_flow["is_sampler"] = is_samper
    name = kwargs_flow.pop("name", "v1")

    if name == "v2":
        n_head = config_backbone["n_head"]
        d_ff = config_backbone["d_ff"]
        ff_type = config_backbone["ff_type"]
        sampler = create_transformer_flow(
            d_model=d_model,
            n_head=n_head,
            d_ff=d_ff,
            ff_type=ff_type,
            **kwargs_flow
        )
    else:
        raise ValueError(f"Unknown sampler model: {name}.")
    return sampler


class ParameterEstimator(nn.Module):
    def __init__(self, backbone, sampler, value_func):
        super().__init__()
        self.backbone = backbone
        self.sampler = sampler
        self.value_func = value_func

    def forward(self, *, param=None, n_draw=None,
                embed_obs=None, embed_sl=None, key_padding_mask=None,
                embed=None, vf_mode=False, recycling=None):
        """This function has the following modes:
            1. embed_obs is not None and embed_sl is not None: Foward
                the backbone to obtain embeddings.

            2. vf_model is True. Compute value function.

            2. n_draw is not None and embed is not None: Draw samples
                and compute log_prob.

            3. param is not None and embed is not None: Estimate log_prob at
                the given points.

        Args:
            param (Tensor): (B, N, D) Spectal line model parameters.

        Returns:
            (Tensor): (B*N, 1) Log prob.
        """
        if embed_obs is not None and embed_sl is not None:
            if recycling is None:
                return self.backbone(
                    embed_obs, embed_sl,
                    key_padding_mask=key_padding_mask,
                )
            return self.backbone(
                embed_obs, embed_sl,
                key_padding_mask=key_padding_mask,
                recycling=recycling,
            )

        if vf_mode:
            if self.value_func is None:
                return None
            # embed (B, L, E)
            # param (B, N, D)
            values = self.value_func(embed)
            return values

        if n_draw is not None and embed is not None:
            # embed (B, L, E)
            batch_size = embed.shape[0]
            embed = embed[:, 0]
            embed = torch.repeat_interleave(embed, n_draw, dim=0) # (B, E) -> (B*N, E)
            samps, log_prob = self.sampler.sample_log_det(embed, embed.device)
            samps = samps.reshape(batch_size, n_draw, -1)
            log_prob = log_prob.reshape(batch_size, n_draw)
            return samps, log_prob # (B, N, D), (B, N)

        if param is not None and embed is not None:
            # embed (B, L, E)
            batch_size = embed.shape[0]
            embed = embed[:, 0]
            embed = torch.repeat_interleave(embed, param.shape[1], dim=0) # (B, E) -> (B*N, E)
            param = torch.flatten(param, end_dim=1)
            log_prob = self.sampler.log_prob(param, embed)
            log_prob = log_prob.reshape(batch_size, -1)
            return log_prob

        raise NotImplementedError

    def draw_samples(self, n_draw, embed_obs, embed_sl,
                     key_padding_mask=None, recycling=0):
        """
        Returns:
            (Tensor): (B, N, D) Samples.
            (Tensor): (B, N) Log probabilty density.
        """
        embed = self(
            embed_obs=embed_obs,
            embed_sl=embed_sl,
            key_padding_mask=key_padding_mask,
            recycling=recycling
        )
        samps, log_prob = self(n_draw=n_draw, embed=embed)
        values = self(
            param=samps.detach(),
            embed=embed,
            vf_mode=True
        )
        if values is not None:
            values = values.squeeze(1)
        return samps, log_prob, values


class ValueFunction(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            SwiGLU(d_model, 128),
            nn.Linear(128, 5),
        )

    def forward(self, embed):
        # embed (B, L, E)
        return self.mlp(embed[:, 1:])