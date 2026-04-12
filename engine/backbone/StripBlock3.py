
import os, sys     
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
   
import warnings  
warnings.filterwarnings('ignore')

import torch     
import torch.nn as nn  
from timm.layers import DropPath

from engine.backbone.conv import Conv

class ConvolutionalGLU(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                      groups=hidden_features),
            act_layer()
        )

        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.conv1x1 = Conv(in_features, out_features, 1) if in_features != out_features else nn.Identity()

    def forward(self, x):

        x_shortcut = self.conv1x1(x)
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x_shortcut + x


class CKConv(nn.Module):
    def __init__(self, c1, c2, kk=[3, 5, 7], s=1):
        super().__init__()

        if not isinstance(kk, list) or not all(ki in [3, 5, 7, 9] for ki in kk):
            raise ValueError("k must be a list containing 3, 5, and/or 7")

        self.kk = kk
        self.c1 = c1
        self.c2 = c2
        self.s = s

        self.branches = nn.ModuleDict()


        for ki in kk:

            self.branches[f'k{ki}_body'] = Conv(c2, c2//2, (3, 3), s=1, g=c2//2)
            self.branches[f'k{ki}_head_h'] = Conv(c2, c2//2, (1, ki), s=s, p=(0, (ki - 1) // 2), g=c2//2)
            self.branches[f'k{ki}_head_v'] = Conv(c2//2, c2//2, (ki, 1), s=s, p=((ki - 1) // 2, 0), g=c2//2)
            self.branches[f'k{ki}_conv2'] = nn.Conv2d(c2//2, c2, 1, groups=c2//2)

        self.conv_fuse = nn.Conv2d(len(kk) * c2, c2, 1, groups=16)   # note 1

    def forward(self, x):

        outputs = []

        for ki in self.kk:
            y = self.branches[f'k{ki}_head_h'](x)
            y = self.branches[f'k{ki}_head_v'](y)
            ys = self.branches[f'k{ki}_body'](x)
            out = ys + y
            out = self.branches[f'k{ki}_conv2'](out)
            outputs.append(out)

        out = torch.cat(outputs, dim=1)
        out = self.conv_fuse(out)

        return out

class MSCK_attn(nn.Module):
    def __init__(self, d_model_r, d_model_o, kk=[3, 5, 7]):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model_r, d_model_o, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = CKConv(d_model_r, d_model_o, kk)  #

    def forward(self, x):
        x_o = self.proj_1(x)
        x_1 = self.activation(x_o)
        x = self.spatial_gating_unit(x_1)
        x = x + x_o
        return x


class MSCK_Block(nn.Module):
    def __init__(self, inc, dim, kk=[3,5,7], mlp_ratio=4., k1=1, k2=11, drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(inc)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = MSCK_attn(inc, dim, kk)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = ConvolutionalGLU(in_features=dim, hidden_features=dim, out_features=dim)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.conv1x1 = Conv(inc, dim, k=1) if inc != dim else nn.Identity()

    def forward(self, x):
        x_o = self.conv1x1(x)
        x_1 = x_o + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x_1 + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x_1)))
        return x

