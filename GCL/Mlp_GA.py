import torch
import torch.nn as nn
from torch.nn.functional import pad
from timm.models.layers import trunc_normal_
from natten.functional import natten2dav, natten2dqkrpb


class GA(nn.Module):  # Gaussian Attention
    """
        Based on Neighborhood Attention 2D Module (https://github.com/SHI-Labs/NATTEN)
    """

    def __init__(
            self,
            dim,
            num_heads,
            kernel_size,
            dilation=1,
            bias=True,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert (
                kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
                dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.proj = nn.Linear(dim, dim)

        k = kernel_size
        self.h_k = (k - 1) // 2
        self.k = k
        self.sigma = 9
        self.sc = 0.1
        self.lr_m = nn.Parameter(torch.ones(k, k) * self.sc * 0.5)

    def _func_gauss(self, attn, feat_x):
        b, h, w, c = feat_x.shape
        device = attn.device

        k = self.k
        crd_ker = torch.linspace(0, k - 1, k).to(device)
        x = crd_ker.view(1, k)
        y = crd_ker.view(k, 1)
        idx_x = self.h_k
        idx_y = self.h_k
        gauss_kernel = torch.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * self.sigma ** 2))
        ker_S = gauss_kernel

        ker_S = ker_S + self.lr_m / self.sc
        attn_gauss = ker_S.view(1, 1, 1, 1, self.k ** 2).repeat(b, self.num_heads, h, w, 1) * attn
        return attn_gauss

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = Hp, Wp
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, H, W, 3, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = natten2dqkrpb(q, k, self.rpb, self.kernel_size, self.dilation)

        attn = self._func_gauss(attn, x)
        attn = attn.softmax(dim=-1)

        x = natten2dav(attn, v, self.kernel_size, self.dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return (
                f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
                + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
                + f"rel_pos_bias={self.rpb is not None}"
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
