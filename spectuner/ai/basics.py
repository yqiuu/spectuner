from torch import nn
from torch.nn import functional as F


def create_ff_net(ff_type, input_size, hidden_size):
    if ff_type == "gelu":
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_size)
        )
    if ff_type == "swiglu":
        return nn.Sequential(
            SwiGLU(input_size, hidden_size),
            nn.Linear(hidden_size, input_size)
        )

    raise ValueError(f"Unknown ff_type: {ff_type}.")


class TransformerLayerPost(nn.Module):
    def __init__(self, d_model, n_head, d_ff, ff_type):
        super().__init__()
        self.atten = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ff_net = create_ff_net(ff_type, d_model, d_ff)
        self.rms_norm_1 = nn.RMSNorm(d_model)
        self.rms_norm_2 = nn.RMSNorm(d_model)

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        x_out = v + self.atten(
            q, k, v,
            need_weights=False,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
        )[0]
        x_out = self.rms_norm_1(x_out)
        x_out = x_out + self.ff_net(x_out)
        x_out = self.rms_norm_2(x_out)
        return x_out


class SwiGLU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 2*hidden_size)

    def forward(self, x_in):
        x_1, x_2 = self.fc(x_in).chunk(2, dim=-1)
        return F.silu(x_1)*x_2