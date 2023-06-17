import torch
from functools import partial

from torch import nn as nn
import torch.nn.functional as F

from iharm.model.modeling.basic_blocks import ConvBlock, GaussianSmoothing
from iharm.model.modeling.unet import UNetEncoder, UNetDecoder
from iharm.model.ops import ChannelAttention
from iharm.model.modeling.lut import LUT
from iharm.model.modeling.refine import Refine
from time import time
import sys

class CDTNet(nn.Module):
    def __init__(
        self,depth,
        norm_layer=nn.BatchNorm2d, batchnorm_from=2,
        attend_from=3, attention_mid_k=2.0,
        image_fusion=False,
        ch=64, max_channels=512,
        backbone_from=-1, backbone_channels=None, backbone_mode='',
        n_lut=4
    ):
        super(CDTNet, self).__init__()
        self.depth = depth
        self.base_resolution = 256
        self.n_lut = n_lut
        self.mean = torch.tensor([.485, .456, .406], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], dtype=torch.float32).view(1, 3, 1, 1)
        self.device = None
        #1.pix2pix
        self.encoder = UNetEncoder(
            depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels, backbone_mode
        )
        self.decoder = UNetDecoder(
            depth, self.encoder.block_channels,
            norm_layer,
            attention_layer=partial(SpatialSeparatedAttention, mid_k=attention_mid_k),
            attend_from=attend_from,
            image_fusion=image_fusion
        )
        #2.rgb2rgb
        self.lut = LUT(256, n_lut, backbone='issam')
        #3.refinement
        self.refine = Refine(feature_channels=32,inner_channel=64)

    def set_resolution(self, hr, lr, finetune_base):
        self.target_resolution = hr
        self.base_resolution = lr
        self.finetune_base = finetune_base

    def init_device(self, input_device):
        if self.device is None:
            self.device = input_device
            self.mean = self.mean.to(self.device)
            self.std = self.std.to(self.device)

    def normalize(self, tensor):
        self.init_device(tensor.device)
        # return self.norm(tensor)
        return (tensor - self.mean) / self.std
         

    def denormalize(self, tensor):
        self.init_device(tensor.device)
        return tensor * self.std + self.mean

    def train(self):
        print("whether finetune the base model:" + str(self.finetune_base))
        if self.finetune_base:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()            
        self.refine.train()
        self.lut.train()
        for param in self.encoder.parameters():
            if self.finetune_base:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in self.decoder.parameters():
            if self.finetune_base:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in self.refine.parameters():
            param.requires_grad = True
        for param in self.lut.parameters():
            param.requires_grad = True

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.refine.eval()
        self.lut.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.refine.parameters():
            param.requires_grad = False
        for param in self.lut.parameters():
            param.requires_grad = False

    def forward(self, image, mask, backbone_features=None):  
        normed_image = self.normalize(image)
        x = torch.cat((normed_image, mask), dim=1)
        basic_input = F.interpolate(x, size=(self.base_resolution,self.base_resolution), mode='bilinear').detach()
        intermediates = self.encoder(basic_input, backbone_features)
        output,output_map = self.decoder(intermediates, basic_input[:,:3,:,:], basic_input[:,3:,:,:])
        lut_output = self.lut(intermediates, image, mask)
        normed_lut = self.normalize(lut_output)
        _, hd_output = self.refine(output, normed_image, mask, output_map, normed_lut, target_resolution=(self.target_resolution,self.target_resolution))
        denormed_hd_output = self.denormalize(hd_output)
        return {'images': denormed_hd_output, 'lut_images':lut_output, 'base_images': self.denormalize(output)}

        
        


class SpatialSeparatedAttention(nn.Module):
    def __init__(self, in_channels, norm_layer, activation, mid_k=2.0):
        super(SpatialSeparatedAttention, self).__init__()
        self.background_gate = ChannelAttention(in_channels)
        self.foreground_gate = ChannelAttention(in_channels)
        self.mix_gate = ChannelAttention(in_channels)

        mid_channels = int(mid_k * in_channels)
        self.learning_block = nn.Sequential(
            ConvBlock(
                in_channels, mid_channels,
                kernel_size=3, stride=1, padding=1,
                norm_layer=norm_layer, activation=activation,
                bias=False,
            ),
            ConvBlock(
                mid_channels, in_channels,
                kernel_size=3, stride=1, padding=1,
                norm_layer=norm_layer, activation=activation,
                bias=False,
            ),
        )
        self.mask_blurring = GaussianSmoothing(1, 7, 1, padding=3)

    def forward(self, x, mask):
        mask = self.mask_blurring(F.interpolate(
            mask, size=x.size()[-2:],
            mode='bilinear', align_corners=True
        ))
        background = self.background_gate(x)
        foreground = self.learning_block(self.foreground_gate(x))
        mix = self.mix_gate(x)
        output = mask * (foreground + mix) + (1 - mask) * background
        return output
