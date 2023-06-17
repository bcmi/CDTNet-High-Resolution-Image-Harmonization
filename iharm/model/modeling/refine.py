import torch
from torch import nn as nn
import torch.nn.functional as F
from iharm.model.modeling.basic_blocks import ConvBlock, DBDownsample, DBUpsample, UpBlock, DownBlock

class Refine(nn.Module):
    def __init__(self,feature_channels = 32,in_channel = 7,inner_channel=32,
                 norm_layer=nn.BatchNorm2d, activation=nn.ELU,image_fusion=True):
        super(Refine, self).__init__()
        self.image_fusion = image_fusion
        self.block = nn.Sequential(
            nn.Conv2d(feature_channels + in_channel, inner_channel, kernel_size=3, stride=1, padding=1),
            norm_layer(inner_channel) if norm_layer is not None else nn.Identity(),
            activation(),
            nn.Conv2d(inner_channel, inner_channel, kernel_size=3, stride=1, padding=1),
            norm_layer(inner_channel) if norm_layer is not None else nn.Identity(),
            activation(),
        )
        if self.image_fusion:
            self.conv_attention = nn.Conv2d(inner_channel, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(inner_channel, 3, 1,1,0)

    def forward(self, ssam_output, comp, mask, ssam_features,lut_output, target_resolution):
        ssam_in = F.interpolate(ssam_output, size=target_resolution, mode='bilinear')
        comp = F.interpolate(comp, size=target_resolution, mode='bilinear')
        mask = F.interpolate(mask, size=target_resolution, mode='bilinear')
        ssam_features = F.interpolate(ssam_features, size=target_resolution, mode='bilinear')
        lut_in = F.interpolate(lut_output, size=target_resolution, mode='bilinear')
        input_1 = torch.cat([ssam_in, lut_in, mask, ssam_features], dim=1)
        output_map = self.block(input_1)

        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(output_map))
            output = attention_map * comp + (1.0 - attention_map) * self.to_rgb(output_map)
        else:
            output = self.to_rgb(output_map)
        return output_map,output



