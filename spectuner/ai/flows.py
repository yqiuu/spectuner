import math

import torch
from torch import nn
from torch.nn import functional as F

from .basics import TransformerLayerPost


def create_transformer_flow(n_dim, n_bin, d_model, n_head, d_ff, ff_type,
                            beta, width_min, permute, n_block, bounds, is_sampler):
    layers = [
        Permute(torch.tensor(permute)),
        create_transformer_coupler(
            n_dim, n_bin, d_model, n_head, d_ff, ff_type, beta, width_min
        )
    ]
    for _ in range(n_block - 1):
        layers.append(ReversePermute(n_dim))
        layers.append(create_transformer_coupler(
            n_dim, n_bin, d_model, n_head, d_ff, ff_type, beta, width_min
        ))
    transform = Sequential(*layers)

    base = StandardUniform(n_dim)
    x_min, x_max = list(zip(*bounds))
    x_min = torch.tensor(x_min)
    x_max = torch.tensor(x_max)
    scaler_rand = Scaler(x_min, x_max)
    return TransformedDistribution(
        base, transform, is_sampler=is_sampler, scaler_rand=scaler_rand
    )


def create_transformer_coupler(n_dim, n_bin, d_model, n_head, d_ff, ff_type,
                               beta, width_min):
    out_size = 2*n_bin + 1
    net = TransformerCoupler(n_dim, out_size, d_model, n_head, d_ff, ff_type)
    coupler = PiecewiseQuadratic(n_dim, n_bin, net, beta, width_min)
    return AutoregressiveCoupling(coupler)


class TransformedDistribution(nn.Module):
    def __init__(self, base, transform, is_sampler=True, scaler_rand=None, scaler_cond=None):
        super(TransformedDistribution, self).__init__()
        self.base = base
        self.transform = transform
        self.is_sampler = is_sampler
        self.scaler_rand = IdenityScaler() if scaler_rand is None else scaler_rand
        self.scaler_cond = IdenityScaler() if scaler_cond is None else scaler_cond

    def log_prob(self, x_rand, x_cond=None):
        x_rand, log_det_jac = self.transform_log_det(x_rand, x_cond)
        log_prob = self.base.log_prob(x_rand) + log_det_jac
        return log_prob

    def transform_samples(self, x_rand, x_cond=None):
        return self.transform_log_det(x_rand, x_cond)[0]

    def transform_log_det(self, x_rand, x_cond):
        if self.is_sampler:
            x_rand = self.scaler_rand.recover(x_rand)
        else:
            x_rand = self.scaler_rand(x_rand)
        x_cond = self.scaler_cond(x_cond)

        log_prob = self.base.log_prob(x_rand)
        if self.transform is None:
            log_prob = self.base.log_prob(x_rand)
            return log_prob

        if self.is_sampler:
            x_rand, log_det_jac = self.transform.reverse(x_rand, x_cond)
        else:
            x_rand, log_det_jac = self.transform(x_rand, x_cond)
        return x_rand, log_det_jac

    def draw_samples(self, num_or_cond, device=None):
        return self.sample_log_det(num_or_cond, device)[0]

    def sample_log_det(self, num_or_cond, device=None):
        if type(num_or_cond) is int:
            num = num_or_cond
            x_cond = None
        else:
            num = len(num_or_cond)
            x_cond = num_or_cond
        x_cond = self.scaler_cond(x_cond)

        samps = self.base.sample(num, device)
        log_prob = self.base.log_prob(samps)
        if self.transform is None:
            return samps, log_prob

        if self.is_sampler:
            samps, log_det_jac = self.transform(samps, x_cond)
            samps = self.scaler_rand.recover(samps)
        else:
            samps, log_det_jac = self.transform.reverse(samps, x_cond)
            samps = self.scaler_rand.recover(samps)
        log_prob = log_prob - log_det_jac
        return samps, log_prob


class StandardUniform(nn.Module):
    def __init__(self, n_dim):
        super(StandardUniform, self).__init__()
        self.n_dim = n_dim

    def log_prob(self, x):
        return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)

    def sample(self, num, device=None):
        return torch.rand(num, self.n_dim, device=device)


class Sequential(nn.Sequential):
    def forward(self, x_rand, x_cond):
        log_det_jac = 0.
        for tm in self.children():
            x_rand, log_det_jac_new = tm(x_rand, x_cond)
            log_det_jac = log_det_jac + log_det_jac_new
        return x_rand, log_det_jac

    def reverse(self, x_rand, x_cond):
        log_det_jac = 0.
        for tm in reversed(tuple(self.children())):
            x_rand, log_det_jac_new = tm.reverse(x_rand, x_cond)
            log_det_jac = log_det_jac + log_det_jac_new
        return x_rand, log_det_jac


class Scaler(nn.Module):
    def __init__(self, x_min, x_max):
        super(Scaler, self).__init__()
        self.register_buffer("x_min", x_min)
        self.register_buffer("scale", x_max - x_min)

    def forward(self, x_in):
        return (x_in - self.x_min)/self.scale

    def recover(self, x_in):
        return self.x_min + self.scale*x_in


class IdenityScaler(nn.Module):
    def forward(self, x_in):
        return x_in

    def recover(self, x_in):
        return x_in


class PiecewiseQuadratic(nn.Module):
    """Piecewise quadratic module."""
    def __init__(self, n_dim, n_bin, net=None, beta=1., width_min=0.):
        super(PiecewiseQuadratic, self).__init__()
        if net is None:
            self.coupler = None
            self.width = nn.Parameter(torch.rand(1, n_dim, n_bin))
            self.v_node = nn.Parameter(torch.randn(1, n_dim, n_bin + 1))
        else:
            self.coupler = net
        self.n_dim = n_dim
        self.n_bin = n_bin
        self.eps_min = 1e-10
        self.eps_max = 1 - self.eps_min
        self.beta = beta
        self.width_min = width_min

    def forward(self, x_in, x_cond=None, **kwargs):
        # x_in (B, D)
        # v_node (B, D, N + 1)
        # width (B, D, N)
        x_in, x_cond = self.clamp(x_in, x_cond)

        width, x_node, y_node, v_0, v_1 = self.derive_params(x_cond, **kwargs)
        x_local = (x_in[..., None] - x_node[..., :-1])/width # (B, D, N)
        x_local[width == 0.] = .5
        cond = self.derive_cond_bin(x_node, x_in)

        x_out = width*(v_0*x_local + .5*(v_1 - v_0)*x_local*x_local) + y_node[..., :-1]
        x_out[cond] = 0.
        x_out = x_out.sum(dim=-1)

        grad = v_0 + (v_1 - v_0)*x_local
        grad[cond] = self.eps_min
        grad = grad.sum(dim=-1)
        grad = torch.clamp(grad, self.eps_min)
        log_det_jac = grad.log().sum(dim=-1, keepdim=True)
        return x_out, log_det_jac

    def reverse(self, x_in, x_cond=None, **kwargs):
        # x_in (B, D)
        x_in, x_cond = self.clamp(x_in, x_cond)

        width, x_node, y_node, v_0, v_1 = self.derive_params(x_cond, **kwargs)
        v_diff = v_1 - v_0
        v_diff[v_diff.abs() < self.eps_min] = self.eps_min
        val = v_0*v_0 + 2.*v_diff*(x_in[..., None] - y_node[..., :-1])/width
        cond = val < 0.
        val[cond] = 0.
        x_local = (torch.sqrt(val) - v_0)/v_diff
        cond = cond | self.derive_cond_bin(y_node, x_in)

        x_out = x_node[..., :-1] + width*x_local
        x_out[cond] = 0.
        x_out = x_out.sum(dim=-1)

        grad = v_0 + v_diff*x_local
        grad[cond] = self.eps_min
        grad = grad.sum(dim=-1)
        grad = torch.clamp(grad, self.eps_min)
        if len(grad.shape) > 1:
            log_det_jac = -grad.log().sum(dim=-1, keepdim=True)
        else:
            log_det_jac = -grad.log().reshape(-1, 1)
        return x_out, log_det_jac

    def derive_params(self, x_cond, **kwargs):
        if self.coupler is None:
            width = self.width
            v_node = self.v_node
        else:
            # v_node (B, D, N + 1)
            # width (B, D, N)
            params = self.call_coupler(x_cond, **kwargs)
            v_node, width = torch.split(params, self.n_bin + 1, dim=-1)

        width = torch.softmax(self.beta*width, dim=-1)
        width = torch.clamp(width, self.width_min)
        width = width/width.sum(dim=-1, keepdim=True)
        x_node = torch.zeros_like(v_node)
        x_node[..., 1:] = torch.cumsum(width, dim=-1)

        y_node = torch.full_like(v_node, -torch.inf)
        tmp = torch.logaddexp(v_node[..., :-1], v_node[..., 1:]) + torch.log(.5*width)
        y_node[..., 1:] = torch.logcumsumexp(tmp, dim=-1)
        y_norm = y_node[..., -1:]
        y_node = torch.exp(y_node - y_norm)
        v_node = torch.exp(v_node - y_norm)
        v_0 = v_node[..., :-1]
        v_1 = v_node[..., 1:]

        return width, x_node, y_node, v_0, v_1

    def derive_cond_bin(self, x_node, x_in):
        inds = torch.searchsorted(x_node, x_in[..., None]).squeeze(dim=-1) - 1
        inds = inds.clamp_(0, self.n_bin - 1)
        cond = F.one_hot(inds, self.n_bin).type(torch.bool)
        return ~cond

    def call_coupler(self, x_cond, **kwargs):
        if 'i_dim' in kwargs:
            params = self.coupler.forward_dimwise(kwargs['i_dim'], kwargs['x_prev'], x_cond)
            return params

        if 'x_rand' in kwargs:
            params = self.coupler(kwargs['x_rand'], x_cond)
        else:
            params = self.coupler(x_cond)
        params = params.reshape(len(params), -1, 2*self.n_bin + 1)
        return params

    def clamp(self, x_in, x_cond):
        x_in = torch.clamp(x_in, self.eps_min, self.eps_max)
        if x_cond is not None:
            x_cond = torch.clamp(x_cond, self.eps_min, self.eps_max)
        return x_in, x_cond


class AutoregressiveCoupling(nn.Module):
    def __init__(self, coupler):
        super(AutoregressiveCoupling, self).__init__()
        self.coupler = coupler

    def forward(self, x_in, x_cond=None):
        return self.coupler(x_in, x_cond, x_rand=x_in)

    def reverse(self, x_in, x_cond=None):
        x_out = torch.zeros_like(x_in)
        log_det_jac = 0.
        for i_dim in range(x_in.shape[-1]):
            # Use clone to avoid inplace operation
            x_prev = x_out[..., :i_dim].clone()
            x_rand = x_in[..., i_dim]
            # The shape of log_grad should be (X, 1)
            x_out[..., i_dim], log_grad \
                = self.coupler.reverse(x_rand, x_cond, i_dim=i_dim, x_prev=x_prev)
            log_det_jac = log_det_jac + log_grad
        return x_out, log_det_jac


class TransformerCoupler(nn.Module):
    """Transformer Coupler.

    This module must work with conditional parameters.
    """
    def __init__(self, n_dim, out_size, d_model, n_head, d_ff, ff_type):
        super().__init__()
        self.lin_1 = nn.Linear(1, d_model, bias=False)
        self.lin_2 = nn.Linear(d_model, out_size)
        self.transformer = TransformerLayerPost(d_model, n_head, d_ff, ff_type)
        self.register_buffer(
            "attn_mask", torch.triu(torch.full((n_dim, n_dim), True), diagonal=1)
        )
        nn.init.normal_(self.lin_2.weight, 0, 0.1/d_model)
        self.marker = nn.Parameter(torch.rand(1, 1, d_model))

    def forward(self, x_in, x_cond):
        # x_in (B, D)
        # x_cond (B, E)
        return self.foward_main(x_in[:, :-1], x_cond)

    def forward_dimwise(self, i_dim, x_in, x_cond):
        # x_in (B, i_dim)
        if i_dim == 0:
            marker = self.marker.repeat(x_in.shape[0], 1, 1)
            x_tmp = marker + x_cond.unsqueeze(1)
            x_out = self.transformer(x_tmp, x_tmp, x_tmp)
            x_out = self.lin_2(x_out.squeeze())
            return x_out

        return self.foward_main(x_in, x_cond)[:, -1]

    def foward_main(self, x_in, x_cond):
        x_out = self.lin_1(x_in.unsqueeze(-1))
        marker = self.marker.repeat(x_in.shape[0], 1, 1)
        x_out = torch.cat([marker, x_out], dim=1) + x_cond.unsqueeze(1)
        seq_len = x_out.shape[1]
        x_out = self.transformer(
            x_out, x_out, x_out,
            attn_mask=self.attn_mask[:seq_len, :seq_len]
        )
        x_out = self.lin_2(x_out)
        return x_out


class Permute(nn.Module):
    def __init__(self, inds_perm):
        super(Permute, self).__init__()
        assert torch.equal(torch.sort(inds_perm).values, torch.arange(len(inds_perm))), \
            "Invalid permute indices."
        inds_list = list(inds_perm)
        inds_back = torch.as_tensor([inds_list.index(idx) for idx in range(len(inds_perm))])
        self.register_buffer('inds_perm', inds_perm)
        self.register_buffer('inds_back', inds_back)

    def forward(self, x_in, x_cond=None):
        return x_in[..., self.inds_perm], 0.

    def reverse(self, x_in, x_cond=None):
        return x_in[..., self.inds_back], 0.


class ReversePermute(Permute):
    def __init__(self, n_dim):
        inds_perm = torch.arange(n_dim - 1, -1, -1)
        super(ReversePermute, self).__init__(inds_perm)