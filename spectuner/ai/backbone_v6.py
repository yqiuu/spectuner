import torch
from torch import nn

from .basics import TransformerLayerPost, SwiGLU


class BackboneV6(nn.Module):
    def __init__(self, d_model, n_head, d_ff, ff_type, n_block,
                 d_sl, in_channels, d_obs, d_patch):
        super().__init__()
        n_patch, rest = divmod(d_obs, d_patch)
        assert rest == 0
        n_patch += 1
        self.mlp = nn.Sequential(
            SwiGLU(d_sl, d_ff),
            nn.Linear(d_ff, d_model)
        )
        self.patching = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=d_patch,
            stride=d_patch
        )
        blocks = nn.ModuleList()
        for _ in range(n_block):
            blocks.append(Block(d_model, n_head, d_ff, ff_type, n_patch))
        self.blocks = blocks
        self.marker_sl = nn.Parameter(torch.randn(1, 1, d_model))
        self.marker_obs = nn.Parameter(torch.randn(1, 1, 1, d_model))

    def forward(self, embed_obs, embed_sl, key_padding_mask=None):
        # embed_obs (B, L, D_obs)
        # embed_sl (B, L, D_sl)
        # key_padding_mask (B, L)
        batch_size = embed_obs.shape[0]
        if key_padding_mask is not None:
            padding = torch.full((batch_size, 1), False, device=embed_obs.device)
            key_padding_mask = torch.cat([padding, key_padding_mask], dim=1)

        # Spectral line block
        x_sl = self.mlp(embed_sl)
        x_sl = torch.cat([self.marker_sl.repeat(batch_size, 1, 1), x_sl], dim=1)

        # Spectrum block
        x_obs = torch.flatten(embed_obs, end_dim=1) # (B*L, C, d_obs)
        x_obs = self.patching(x_obs) # (B*L, d_model, P)
        x_obs = torch.transpose(x_obs, 1, 2) # (B*L, P, d_model)
        x_obs = torch.unflatten(x_obs, 0, (batch_size, -1)) # (B, L, P, d_model)
        x_obs = torch.cat([self.marker_obs.repeat(batch_size, x_obs.shape[1], 1, 1), x_obs], dim=2)
        x_obs = torch.cat([self.marker_obs.repeat(batch_size, 1, x_obs.shape[2], 1), x_obs], dim=1)
        if key_padding_mask is not None:
            key_padding_mask_2 = key_padding_mask.repeat_interleave(x_obs.shape[2], dim=0)
        else:
            key_padding_mask_2 = None

        # Main
        for block in self.blocks:
            x_sl, x_obs = block(x_sl, x_obs, key_padding_mask, key_padding_mask_2)
        return x_sl


class Block(nn.Module):
    def __init__(self, d_model, n_head, d_ff, ff_type, n_patch):
        super().__init__()
        self.t_sl_1 = TransformerLayerPost(d_model, n_head, d_ff, ff_type)
        self.t_sl_2 = TransformerLayerPost(d_model, n_head, d_ff, ff_type)
        self.t_obs = Transformer3D(d_model, n_head, d_ff, ff_type)
        self.query_1 = nn.Sequential(
            SwiGLU(d_model, n_patch*d_model),
            nn.Unflatten(1, (n_patch, d_model))
        )
        self.query_2 = SwiGLU(d_model, d_model)

    def forward(self, x_sl, x_obs, mask_sl=None, mask_obs=None):
        # x_sl (B, L, d_model)
        # x_obs (B, L, P, d_model)
        q_1 = self.query_1(x_sl[:, 0])
        q_2 = self.query_2(x_sl)
        x_obs = self.t_obs(x_obs, q_1, q_2, mask_obs)
        x_sl = x_sl + x_obs[:, :, 0]
        x_sl = self.t_sl_1(x_sl, x_sl, x_sl, mask_sl)
        x_sl = self.t_sl_2(x_sl, x_sl, x_sl, mask_sl)
        return x_sl, x_obs


class Transformer3D(nn.Module):
    def __init__(self, d_model, n_head, d_ff, ff_type):
        super().__init__()
        self.atten = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.rms_norm = nn.RMSNorm(d_model)
        self.transformer = TransformerLayerPost(d_model, n_head, d_ff, ff_type)

    def forward(self, x_in, q_1, q_2, key_padding_mask=None):
        # x_in (B, L, P, d_model)
        # q_1 (B, P, d_model)
        # q_2 (B, L, d_model)
        batch_size, seq_len, patch_len, _ = x_in.shape

        x_out = torch.flatten(x_in, end_dim=1) # (B*L, P, d_model)
        q_1 = q_1.unsqueeze(1)
        q_1 = torch.repeat_interleave(q_1, seq_len, dim=1).flatten(end_dim=1)
        x_out = x_out + self.atten(q_1, x_out, x_out, need_weights=False)[0]
        x_out = self.rms_norm(x_out)
        x_out = torch.unflatten(x_out, 0, sizes=(batch_size, seq_len)) # (B, L, P, d_model)

        x_out = torch.transpose(x_out, 1, 2) # (B, P, L, d_model)
        x_out = torch.flatten(x_out, end_dim=1) # (B*P, L, d_model)
        q_2 = q_2.unsqueeze(1)
        q_2 = torch.repeat_interleave(q_2, patch_len, dim=1).flatten(end_dim=1)
        x_out = self.transformer(q_2, x_out, x_out, key_padding_mask=key_padding_mask)
        x_out = torch.unflatten(x_out, 0, sizes=(batch_size, -1))  # (B, P, L, d_model)
        x_out = torch.transpose(x_out, 1, 2) # (B, L, P, d_model)
        return x_out