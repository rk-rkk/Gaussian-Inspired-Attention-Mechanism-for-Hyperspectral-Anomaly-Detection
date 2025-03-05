import torch
from timm.models.layers import trunc_normal_
import torch.nn as nn

from GCL.Mlp_GA import Mlp, GA


class GCLBlock(nn.Module):
    def __init__(self, dim, num_heads, sc=1.,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.drop_path = nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.nat = GA(dim=dim, kernel_size=13, dilation=3, num_heads=num_heads)

    def _func_attn(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # B H W C
        x = self.nat(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, *inputs):
        """ Forward function.
            input tensor size (B, H*W, C).
        """
        x, feat_shape = inputs
        B, C, H, W = feat_shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # B C H W

        x = self._func_attn(x)
        x = x.contiguous().view(B, C, H * W).permute(0, 2, 1).contiguous()  # B HW C

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class GCL(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 depth=1,
                 num_head=2,
                 window_size=7,
                 neig_win_num=1,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False):
        super().__init__()

        self.num_feature = embed_dim
        self.num_head = num_head

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # build blocks
        self.blocks = nn.ModuleList([
            GCLBlock(
                dim=self.num_feature,
                num_heads=self.num_head,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(self.num_feature)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        x = inputs.permute(0, 2, 3, 1).contiguous()

        x = x.view(B, H * W, C)
        for f in self.blocks:
            x = f(x, inputs.shape)

        x = self.norm(x)
        out = x.view(-1, H, W, self.num_feature).permute(0, 3, 1, 2).contiguous()
        return out
