import torch.nn as nn
import numpy as np
import torch



def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class DepthWiseConv(nn.Module):
    def __init__(self, inc, kernel_size, stride=1):
        super().__init__()
        padding = 1
        if kernel_size == 1:
            padding = 0
        self.conv = conv_bn(inc, inc, kernel_size, stride, padding, inc)

    def forward(self, x):
        return self.conv(x)


# https://arxiv.org/abs/2206.04040
# unofficial: https://github.com/shoutOutYangJie/MobileOne
class PointWiseConv(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = conv_bn(inc, outc, 1, 1, 0)

    def forward(self, x):
        return self.conv(x)


class MobileOneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k,
                 stride=1, dilation=1, padding_mode='zeros', deploy=False, use_se=False):
        super(MobileOneBlock, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deploy = deploy
        kernel_size = 3
        padding = 1
        assert kernel_size == 3
        assert padding == 1
        self.k = k
        padding_11 = padding - kernel_size // 2
        self.nonlinearity = nn.ReLU()

        if use_se:
            # self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
            ...
        else:
            self.se = nn.Identity()

        if deploy:
            self.dw_reparam = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding, dilation=dilation, groups=in_channels, bias=True,
                                        padding_mode=padding_mode)
            self.deploy_bn = nn.BatchNorm2d(in_channels)
            self.pw_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                        bias=True)

        else:
            self.dw_bn_layer = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            for k_idx in range(k):
                setattr(self, f'dw_3x3_{k_idx}',
                        DepthWiseConv(in_channels, 3, stride=stride)
                        )
            self.dw_1x1 = DepthWiseConv(in_channels, 1, stride=stride)

            self.pw_bn_layer = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            for k_idx in range(k):
                setattr(self, f'pw_1x1_{k_idx}',
                        PointWiseConv(in_channels, out_channels)
                        )

    def forward(self, inputs):
        if self.deploy:
            x = self.dw_reparam(inputs)
            x = self.deploy_bn(x)
            x = self.nonlinearity(x)
            x = self.pw_reparam(x)
            x = self.deploy_bn(x)
            x = self.nonlinearity(x)
            return x

        if self.dw_bn_layer is None:
            id_out = 0
        else:
            id_out = self.dw_bn_layer(inputs)

        x_conv_3x3 = []
        for k_idx in range(self.k):
            x = getattr(self, f'dw_3x3_{k_idx}')(inputs)
            x_conv_3x3.append(x)
        x_conv_1x1 = self.dw_1x1(inputs)

        x = id_out + x_conv_1x1 + sum(x_conv_3x3)
        x = self.nonlinearity(self.se(x))

        # 1x1 conv # https://github.com/iscyy/yoloair
        if self.pw_bn_layer is None:
            id_out = 0
        else:
            id_out = self.pw_bn_layer(x)
        x_conv_1x1 = []
        for k_idx in range(self.k):
            x_conv_1x1.append(getattr(self, f'pw_1x1_{k_idx}')(x))
        x = id_out + sum(x_conv_1x1)
        x = self.nonlinearity(x)
        return x

class MobileOne(nn.Module):
    # MobileOne
    def __init__(self, in_channels, out_channels, n, k,
                 stride=1, dilation=1, padding_mode='zeros', deploy=False, use_se=False):
        super().__init__()
        self.m = nn.Sequential(*[MobileOneBlock(in_channels, out_channels, k, stride, deploy) for _ in range(n)])

    def forward(self, x):
        x = self.m(x)
        return x




