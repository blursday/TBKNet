import torch
import torch.nn as nn
import torch.nn.functional as F

#from tbknet.nn.fasterNeXt import PConv


class LayerNorms(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class FocalNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.dwconv_3 = nn.Conv2d(dim, dim, kernel_size=7, padding='same', groups=dim, dilation=3)  # depthwise conv

        self.norm = LayerNorms(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1 = x + self.drop_path2(input)
        x = self.dwconv_3(x1)
        x = x + self.drop_path3(x1)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

class PConv(nn.Module):
    # PWConv Block
    def __init__(self,
                 dim=int,
                 n_div=int, # 设置为4
                 forward= "split_cat",
                 kernel_size=3,
                 g=True): # g=True为调用PWconv, g=False为PConv
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv
        self.g1 = int(dim / 4) if g is True else 1
        self.conv=nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size, stride=1,padding=(kernel_size -1) // 2, groups = self.g1, bias=False)
        if forward == "slicing":
            self.forward = self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError
    def forward_slicing(self, x):
        x[:, :self.dim_conv, :, :] = self.conv(x[:, :self.dim_conv, :, :])
        return x

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x,[self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        x=torch.cat((x1,x2),1)
        return x


class FocalPWConvBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.dwconv_3 = nn.Conv2d(dim, dim, kernel_size=7, padding='same', groups=dim, dilation=3)  # depthwise conv

        self.norm = LayerNorms(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        #self.act = nn.ReLU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.pconv = PConv(dim, 2, "split_cat", 3)

    def forward(self, x):
        input = x
        x = self.pconv(x)
        # x = self.dwconv(x)
        x = x + self.drop_path2(input)
        ## x =self.pconv(x1)
        # x = self.dwconv_3(x1)
        ## x = x + self.drop_path3(x1)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

class FocalPWConvBlock1(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.dwconv_3 = nn.Conv2d(dim, dim, kernel_size=7, padding='same', groups=dim, dilation=3)  # depthwise conv

        self.norm = LayerNorms(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        #self.act = nn.ReLU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.pconv =PConv(dim, 2, "split_cat", 3)
        #self.pconv = PConv(dim, 2, "split_cat", 7)
        #self.conv = Conv(dim, dim*4, 3, 1, 1)
        self.conv = nn.Conv2d(dim, dim * 4, kernel_size=3, stride=1,padding= 1)

    def forward(self, x):
        input = x
        x = self.pconv(x)
        #x = self.dwconv(x)
        x = x + self.drop_path2(input)
       # x =self.pconv(x1)
        #x = self.dwconv_3(x1)
       ## x = x + self.drop_path3(x1)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        #x = self.conv(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

class FocalPWConvBlock2(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.dwconv_3 = nn.Conv2d(dim, dim, kernel_size=7, padding='same', groups=dim, dilation=3)  # depthwise conv

        self.norm = LayerNorms(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim,dim)
        self.act = nn.GELU()
        #self.act = nn.ReLU()
        self.pwconv2 = nn.Linear( dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.pconv =PConv(dim, 2, "split_cat", 3)
        #self.pconv = PConv(dim, 2, "split_cat", 7)
        #self.conv = Conv(dim, dim*4, 3, 1, 1)
        self.conv = nn.Conv2d(dim, dim * 4, kernel_size=3, stride=1,padding= 1)

    def forward(self, x):
        input = x
        x = self.pconv(x)
        #x = self.dwconv(x)
        x = x + self.drop_path2(input)
       # x =self.pconv(x1)
        #x = self.dwconv_3(x1)
       ## x = x + self.drop_path3(x1)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        #x = self.conv(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x
