import math
import numbers

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.utils import spectral_norm

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        kernel_size=4, stride=2, padding=1,
        norm_layer=nn.BatchNorm2d, activation=nn.ELU,
        bias=True,
    ):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            activation(),
        )
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        # self.norm = norm_layer(out_channels) if norm_layer is not None else nn.Identity()
        # self.act = activation()

    def forward(self, x):
        # x = self.conv(x)
        # x = self.norm(x)
        # x = self.act(x)
        return self.block(x)

class DeconvBlock(nn.Module):
    def __init__(self, 
        in_channels, out_channels, 
        kernel_size=4, stride=2, padding=1, 
        norm_layer=nn.BatchNorm2d, activation=nn.PReLU, 
        bias=True,
    ):
        super(DeconvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            activation(),
        )
        # self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        # self.norm = norm_layer(out_channels) if norm_layer is not None else nn.Identity()
        # self.act = activation()

    def forward(self, x):
        # x = self.deconv(x)
        # x = self.norm(x)
        # x = self.act(x)
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm_layer, kernel_size=3, padding=1, activation=nn.ReLU(True), use_dropout=False):
        super(ResBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim,kernel_size,padding, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, kernel_size, padding, norm_layer, activation, use_dropout):
        if isinstance(padding, tuple):
            padding = (padding[1],padding[1], padding[0],padding[0])
        conv_block = []
        conv_block += [
                    # norm_layer(dim) if norm_layer is not None else nn.Identity(),
                    # activation,
                    nn.ReplicationPad2d(padding),
                    nn.Conv2d(dim, dim, kernel_size=kernel_size),
                    norm_layer(dim) if norm_layer is not None else nn.Identity(),
                    activation,
                       ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [
                    # norm_layer(dim) if norm_layer is not None else nn.Identity(),
                    # activation,
                    nn.ReplicationPad2d(padding),
                    nn.Conv2d(dim, dim, kernel_size=kernel_size),
                    norm_layer(dim) if norm_layer is not None else nn.Identity(),
                    activation,
                    ]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class SepConvHead(nn.Module):
    def __init__(self, num_outputs, in_channels, mid_channels, num_layers=1,
                 kernel_size=3, padding=1, dropout_ratio=0.0, dropout_indx=0,
                 norm_layer=nn.BatchNorm2d):
        super(SepConvHead, self).__init__()

        sepconvhead = []

        for i in range(num_layers):
            sepconvhead.append(
                SeparableConv2d(in_channels=in_channels if i == 0 else mid_channels,
                                out_channels=mid_channels,
                                dw_kernel=kernel_size, dw_padding=padding,
                                norm_layer=norm_layer, activation='relu')
            )
            if dropout_ratio > 0 and dropout_indx == i:
                sepconvhead.append(nn.Dropout(dropout_ratio))

        sepconvhead.append(
            nn.Conv2d(in_channels=mid_channels, out_channels=num_outputs, kernel_size=1, padding=0)
        )

        self.layers = nn.Sequential(*sepconvhead)

    def forward(self, *inputs):
        x = inputs[0]

        return self.layers(x)


def select_activation_function(activation):
    if isinstance(activation, str):
        if activation.lower() == 'relu':
            return nn.ReLU
        elif activation.lower() == 'softplus':
            return nn.Softplus
        else:
            raise ValueError(f"Unknown activation type {activation}")
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise ValueError(f"Unknown activation type {activation}")


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, dw_kernel, dw_padding, dw_stride=1,
                 activation=None, use_bias=False, norm_layer=None):
        super(SeparableConv2d, self).__init__()
        _activation = select_activation_function(activation)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=dw_kernel, stride=dw_stride,
                      padding=dw_padding, bias=use_bias, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            _activation()
        )

    def forward(self, x):
        return self.body(x)




class GaussianSmoothing(nn.Module):
    """
    https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
    Apply gaussian smoothing on a tensor (1d, 2d, 3d).
    Filtering is performed seperately for each channel in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors.
            Output will have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data. Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, padding=0, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = 1.
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, grid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2.
            kernel *= torch.exp(-((grid - mean) / std) ** 2 / 2) / (std * (2 * math.pi) ** 0.5)
        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight.
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = torch.repeat_interleave(kernel, channels, 0)

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.padding = padding

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, padding=self.padding, groups=self.groups)


class MaxPoolDownSize(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, depth):
        super(MaxPoolDownSize, self).__init__()
        self.depth = depth
        self.reduce_conv = ConvBlock(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.convs = nn.ModuleList([
            ConvBlock(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
            for conv_i in range(depth)
        ])
        self.pool2d = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        outputs = []

        output = self.reduce_conv(x)

        for conv_i, conv in enumerate(self.convs):
            output = output if conv_i == 0 else self.pool2d(output)
            outputs.append(conv(output))

        return outputs


class UpPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks=3, st=1, padding=1, scale_factor=2, norm='none', activation='relu', pad_type='zero', use_bias=True, activation_first=False):
        super(UpPBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'sn':
            self.norm = nn.Identity()
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.ModuleList([
            nn.Conv2d(in_dim, out_dim * (scale_factor**2), ks, st, bias=self.use_bias),
            nn.PixelShuffle(scale_factor)]  
        )
        if norm == 'sn':
            self.conv = nn.Sequential(*[
            spectral_norm(nn.Conv2d(in_dim, out_dim * (scale_factor**2), ks, st, bias=self.use_bias)),
            nn.PixelShuffle(scale_factor)
        ])

    def forward(self, x):
        if self.activation_first:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x

class DBUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, ks=4, st=2, padding=1, bias=True, activation='relu', norm='bn',activation_first=False):
        super(DBUpsample, self).__init__()
        padding = (ks - 1) // 2
        ngf = out_channel
        self.up_conv1 = UpPBlock(in_channel, ngf, scale_factor=2, norm=norm, use_bias=bias, activation=activation, activation_first=activation_first)
        self.down_conv1 = Conv2dBlock(ngf, ngf, ks,st,padding, norm=norm, use_bias=bias, activation=activation, activation_first=activation_first)
        self.up_conv2 = UpPBlock(ngf, ngf, scale_factor=2, norm=norm, use_bias=bias, activation=activation, activation_first=activation_first)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.down_conv1(h0)
        h1 = self.up_conv2(l0 - x)
        return h0 + h1

class DBDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, ks=4, st=2, padding=1, bias=True, activation='relu', norm='bn', activation_first=False):
        super(DBDownsample, self).__init__()
        padding = (ks - 1) // 2
        ngf = out_channel
        if in_channel != out_channel:
            self.in_conv = Conv2dBlock(in_channel, out_channel, 1,1,0, norm='none', activation=activation, use_bias=bias, activation_first=activation_first)
        else: self.in_conv = None

        self.down_conv1 = Conv2dBlock(ngf, ngf, ks, st, padding, norm=norm, activation=activation, use_bias=bias, activation_first=activation_first)
        self.up_conv1 = UpPBlock(ngf, ngf, scale_factor=2, norm=norm, use_bias=bias, activation=activation, activation_first=activation_first)
        self.down_conv2 = Conv2dBlock(ngf, ngf, ks, st, padding, norm=norm, activation=activation, use_bias=bias, activation_first=activation_first)

    def forward(self, x):
        if self.in_conv:
            x  = self.in_conv(x)
        l0 = self.down_conv1(x)
        h0 = self.up_conv1(l0)
        l1 = self.down_conv1(h0 - x)
        return l0 + l1



class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'sn':
            self.norm = nn.Identity()
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)
        if norm == 'sn':
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        if self.activation_first:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation=nn.PReLU, norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, norm_layer=None, activation=activation)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, norm_layer=None, activation=activation)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, norm_layer=None, activation=activation)        

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation=nn.PReLU, norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, norm_layer=None, activation=activation)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, norm_layer=None, activation=activation)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, norm_layer=None, activation=activation)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0