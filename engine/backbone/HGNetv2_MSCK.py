
import torch     
import torch.nn as nn
import torch.nn.functional as F
from ..core import register

from .hgnetv2 import HGNetv2, HG_Stage
from .StripBlock3 import MSCK_Block
from .conv import Conv

kaiming_normal_ = nn.init.kaiming_normal_     
zeros_ = nn.init.zeros_     
ones_ = nn.init.ones_  
 
__all__ = ['HGNetv2_MSCK']

class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=3,
            stride=1,
            groups=1,

    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_chs)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class CKConv_c(nn.Module):
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
            self.branches[f'k{ki}_body'] = Conv(c2, c2, (3, 3), s=1, g=c2)
            self.branches[f'k{ki}_head_h'] = Conv(c2, c2, (1, ki), s=s, p=(0, (ki - 1) // 2), g=c2)
            self.branches[f'k{ki}_head_v'] = Conv(c2, c2, (ki, 1), s=s, p=((ki - 1) // 2, 0), g=c2)
            self.branches[f'k{ki}_conv2'] = nn.Conv2d(c2, c2, 1, groups=c2)

        self.conv_fuse = nn.Conv2d(len(kk) * c2, c2, 1, groups=16)

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


class StemBlock_ck(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_chs,
            mid_chs,
            kernel_size=3,
            stride=2,
        )
        self.stem2a = ConvBNAct(
            mid_chs,
            mid_chs // 2,
            kernel_size=2,
            stride=1,
        )
        self.stem2b = ConvBNAct(
            mid_chs // 2,
            mid_chs,
            kernel_size=2,
            stride=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

        self.stem3 = ConvBNAct(
            mid_chs * 2,
            out_chs,
            kernel_size=3,
            stride=2,
        )

        self.stem5 = CKConv_c(
            out_chs,
            out_chs,
            kk=[3, 5, 7],
        )
    def forward(self, x):

        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (1, 0, 1, 0))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem5(x)

        return x

class MSCK_Stage(HG_Stage):
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num, downsample=True, light_block=False, kernel_size=3, use_lab=False, agg='se', drop_path=0):
        super().__init__(in_chs, mid_chs, out_chs, block_num, layer_num, downsample, light_block, kernel_size, use_lab, agg, drop_path)     

        blocks_list = []
        ks = [int(d) for d in str(kernel_size)]
        for i in range(block_num):
            blocks_list.append(
                MSCK_Block(
                    in_chs if i == 0 else out_chs,
                    out_chs,
                    kk=ks,
                )
            )

        self.blocks = nn.Sequential(*blocks_list)
 
@register()    
class HGNetv2_MSCK(HGNetv2):

    arch_configs = {
        'B1_0': {
            'stem_channels': [3, 16, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 2, True, False, 5, 3],
                "stage3": [256, 64, 320, 2, True, True, 5, 3],
                "stage4": [320, 128, 512, 1, True, True, 7, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B1_1': {
            'stem_channels': [3, 16, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 2, True, False, 5, 3],
                "stage3": [256, 64, 320, 3, True, True, 5, 3],
                "stage4": [320, 128, 512, 2, True, True, 7, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B1_2': {
            'stem_channels': [3, 16, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 128, 1, False, False, 3, 3],
                "stage2": [128, 64, 256, 2, True, False, 5, 3],
                "stage3": [256, 64, 320, 3, True, True, 5, 3],
                "stage4": [320, 128, 512, 2, True, True, 7, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
    }
    def __init__(self, name, use_lab=False, return_idx=..., freeze_stem_only=True, freeze_at=0, freeze_norm=True, pretrained=True, agg='se', local_model_dir='weight/hgnetv2/'):    
        super().__init__(name, use_lab, return_idx, freeze_stem_only, freeze_at, freeze_norm, pretrained, agg, local_model_dir)   
    
        stage_config = self.arch_configs[name]['stage_config']

        stem_channels = self.arch_configs[name]['stem_channels']
        # stem
        self.stem = StemBlock_ck(
                in_chs=stem_channels[0],
                mid_chs=stem_channels[1],
                out_chs=stem_channels[2],
                use_lab=use_lab)

        # stages
        self.stages = nn.ModuleList() 
        for i, k in enumerate(stage_config):  
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[
                k]     
            self.stages.append(
                MSCK_Stage(
                    in_channels, 
                    mid_channels, 
                    out_channels, 
                    block_num,
                    layer_num,     
                    downsample,  
                    light_block, 
                    kernel_size,
                    use_lab,
                    agg))    
