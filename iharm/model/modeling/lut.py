import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import numpy as np
import math
import trilinear
import cv2
import sys

def initial_luts(lut_num=3):
    lut_list=[]
    lut_list.append(Generator3DLUT_identity())
    for ii in range(lut_num-1):
        lut_list.append(Generator3DLUT_zero())
    return lut_list


def weights_init_normal_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Weight_predictor_idih(nn.Module):
    def __init__(self, in_channels=1024,out_channels=3):
        super(Weight_predictor_idih, self).__init__()

        # self.dp = nn.Dropout(p=0.5)
        self.mid_ch = 256
        self.conv = nn.Conv2d(in_channels, self.mid_ch, 1, 1, padding=0)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(self.mid_ch*2, out_channels, 1, 1, padding=0)

                    
    def forward(self, encoder_outputs, mask):
        fea0 = encoder_outputs[0]
        fea1 = encoder_outputs[1]
        fea2 = encoder_outputs[2]
        up_fea0 = F.interpolate(fea0, size=fea2.shape[2:], mode='bilinear')
        up_fea1 = F.interpolate(fea1, size=fea2.shape[2:], mode='bilinear')
        fea_input = torch.cat((up_fea0, up_fea1, fea2),1)
        x = self.conv(fea_input)
        down_mask = F.interpolate(mask, size=fea2.shape[2:], mode='bilinear')
        fg_feature = self.avg_pooling(x*down_mask)
        bg_feature = self.avg_pooling(x*(1-down_mask))
        fgbg_fea = torch.cat((fg_feature, bg_feature),1)
        x = self.fc(fgbg_fea)

        return x
    
class Weight_predictor_issam(nn.Module):
    def __init__(self, in_channels=256, out_channels=3, fb=True):
        super(Weight_predictor_issam, self).__init__()

        # self.dp = nn.Dropout(p=0.5)
        self.mid_ch = 256
        self.fb = fb
        self.conv = nn.Conv2d(in_channels, self.mid_ch, 1, 1, padding=0)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self.fb:
            # print(self.mid_ch*2)
            self.fc = nn.Conv2d(self.mid_ch*2, out_channels, 1, 1, padding=0)
        else:
            # print(self.mid_ch)
            self.fc = nn.Conv2d(self.mid_ch, out_channels, 1, 1, padding=0)

                    
    def forward(self, encoder_outputs, mask):
        # fea0 = encoder_outputs[0]
        # fea1 = encoder_outputs[1]
        # fea2 = encoder_outputs[2]
        # up_fea0 = F.interpolate(fea0, size=fea2.shape[2:], mode='bilinear')
        # up_fea1 = F.interpolate(fea1, size=fea2.shape[2:], mode='bilinear')
        # fea_input = torch.cat((up_fea0, up_fea1, fea2),1)
        fea_input = encoder_outputs[0]
        # print('fea shape', fea_input.shape)
        x = self.conv(fea_input)
        if self.fb:
            down_mask = F.interpolate(mask, size=fea_input.shape[2:], mode='bilinear')
            # print(torch.unique(down_mask))
            fg_feature = self.avg_pooling(x*down_mask)
            bg_feature = self.avg_pooling(x*(1-down_mask))
            fgbg_fea = torch.cat((fg_feature, bg_feature),1)
            x = self.fc(fgbg_fea)
        else:
            feature = self.avg_pooling(x)
            x = self.fc(feature)
        return x


    
class Weight_predictor_issam_w_local(nn.Module):
    def __init__(self, in_channels=256, out_channels=3, patch_size=8, concate=True):
        super(Weight_predictor_issam_w_local, self).__init__()

        self.mid_ch = 256
        self.patch_size = patch_size
        self.fea_patches = []
        self.mask_patches = []
        self.concate = concate
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, self.mid_ch, 1, 1, padding=0)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self.concate:
            self.fc1 = nn.Conv2d(self.mid_ch*4, self.mid_ch*2, 1, 1, padding=0)
        self.fc2 = nn.Conv2d(self.mid_ch*2, self.out_channels, 1, 1, padding=0)

    def to_patch(self, tensor):
        list = []
        h,w = tensor.shape[-2:]
        n_x = int(h/self.patch_size)
        n_y = int(w/self.patch_size)
        # print('tensor h w nx ny',h,w,n_x,n_y)
        list.append([tensor] * (n_x * n_y))
        # print('len(list)',len(list[0]))
        for x in range(n_x):
            top = x * self.patch_size
            for y in range(n_y):
                left = y * self.patch_size
                one_patch = tensor[:,:,top:top+self.patch_size, left:left+self.patch_size]
                list[0][x * n_y + y] = one_patch
        return list, int(n_x), int(n_y)

    def forward(self, encoder_outputs, mask):
        fea_input = encoder_outputs[0]
        down_mask = F.interpolate(mask, size=fea_input.shape[2:], mode='bilinear')
        self.fea_patches, n_x, n_y = self.to_patch(fea_input)
        self.mask_patches,_,_ = self.to_patch(down_mask)
        step_x = int(mask.shape[2]/n_x)
        step_y = int(mask.shape[3]/n_y)
        ori_weight_map = torch.FloatTensor(np.zeros((1, self.out_channels, n_x, n_y))).cuda()
        output_weight_map = torch.FloatTensor(np.zeros((1, self.out_channels, mask.shape[2], mask.shape[3]))).cuda()
        # print(output_weight_map.shape,output_weight_map.device)
        ##global
        x = self.conv(fea_input)
        fg_feature = self.avg_pooling(x*down_mask)
        bg_feature = self.avg_pooling(x*(1-down_mask))
        global_fgbg_fea = torch.cat((fg_feature, bg_feature),1)
        ##local
        for x in range(n_x):
            top = x * step_x
            for y in range(n_y):
                left = y * step_y
                local_fg_feature = self.avg_pooling(self.fea_patches[0][x * n_y + y]*self.mask_patches[0][x * n_y + y])
                local_bg_feature = self.avg_pooling(self.fea_patches[0][x * n_y + y]*self.mask_patches[0][x * n_y + y])
                local_fgbg_fea = torch.cat((local_fg_feature, local_bg_feature),1)
                out=torch.FloatTensor(np.zeros((1,self.out_channels,1,1))).cuda()
                if self.concate:
                    gl_fgbg_fea = torch.cat((global_fgbg_fea, local_fgbg_fea),1)
                    out=self.fc1(gl_fgbg_fea)
                    out=self.fc2(out)
                else:
                    if (True in (self.mask_patches[0][x * n_y + y]>0)) and (True in (self.mask_patches[0][x * n_y + y]==0)):
                        gl_fgbg_fea = (global_fgbg_fea + local_fgbg_fea)/2
                        out=self.fc2(gl_fgbg_fea)
                    elif (True in (self.mask_patches[0][x * n_y + y]>0)) and not (True in (self.mask_patches[0][x * n_y + y]==0)):
                        gl_fgbg_fea = global_fgbg_fea
                        out=self.fc2(gl_fgbg_fea)
                    else:
                        out=out
                ori_weight_map[:,:,x:x+1,y:y+1] = out
                patch_out = out.expand([1,self.out_channels,step_x,step_y])
                # print('fea x y ii', x,y,x * n_y + y)
                # print('pos',top,top+step_x,left,left+step_y)
                output_weight_map[:,:,top:top+step_x,left:left+step_y] += patch_out
        # output_weight_map = output_weight_map*mask
        return output_weight_map, ori_weight_map

class LUT_w_local(nn.Module):
    def __init__(self, in_channels=1024, n_lut=3, backbone='issam', patch_size=8, concate=True):
        super(LUT_w_local, self).__init__()
        self.n_lut = n_lut
        if self.n_lut==4:
            self.LUT0 = Generator3DLUT_identity()
            self.LUT1 = Generator3DLUT_zero()
            self.LUT2 = Generator3DLUT_zero()
            self.LUT3 = Generator3DLUT_zero()
        self.backbone=backbone
        if self.backbone=='issam':
            self.classifier = Weight_predictor_issam_w_local(in_channels, n_lut, patch_size, concate)
    def forward(self, encoder_outputs, image, mask):
        pred_weights, ori_weight_map = self.classifier(encoder_outputs, mask) #1 4 1024 1024
        if len(pred_weights.shape) == 1:
            pred_weights = pred_weights.unsqueeze(0)
        combine_A = image.new(image.size())
        if self.n_lut==4:
            gen_A0 = self.LUT0(image)
            gen_A1 = self.LUT1(image)
            gen_A2 = self.LUT2(image)
            gen_A3 = self.LUT3(image)
            for b in range(image.size(0)):
                # pred_weights==0
                combine_A[b,:,:,:] = pred_weights[b,0,:,:] * gen_A0[b,:,:,:] + pred_weights[b,1,:,:] * gen_A1[b,:,:,:]\
                     + pred_weights[b,2,:,:] * gen_A2[b,:,:,:] + pred_weights[b,3,:,:] * gen_A3[b,:,:,:] #+ pred_weights[b,4] * gen_A4[b,:,:,:]
        combine_A = combine_A*mask+image*(1-mask)

        return combine_A, pred_weights, ori_weight_map


class LUT(nn.Module):
    def __init__(self, in_channels=1024, n_lut=3, backbone='issam', fb=True, clamp=False):
        super(LUT, self).__init__()
        self.n_lut = n_lut
        self.fb = fb
        self.clamp = clamp
        # print(self.fb)
        
        if self.n_lut==3:
            self.LUT0 = Generator3DLUT_identity()
            self.LUT1 = Generator3DLUT_zero()
            self.LUT2 = Generator3DLUT_zero()
        elif self.n_lut==2:
            self.LUT0 = Generator3DLUT_identity()
            self.LUT1 = Generator3DLUT_zero()
        elif self.n_lut==1:
            self.LUT0 = Generator3DLUT_identity()
        elif self.n_lut==4:
            self.LUT0 = Generator3DLUT_identity()
            self.LUT1 = Generator3DLUT_zero()
            self.LUT2 = Generator3DLUT_zero()
            self.LUT3 = Generator3DLUT_zero()

        elif self.n_lut==5:
            self.LUT0 = Generator3DLUT_identity()
            self.LUT1 = Generator3DLUT_zero()
            self.LUT2 = Generator3DLUT_zero()
            self.LUT3 = Generator3DLUT_zero()
            self.LUT4 = Generator3DLUT_zero()
        elif self.n_lut==6:
            self.LUT0 = Generator3DLUT_identity()
            self.LUT1 = Generator3DLUT_zero()
            self.LUT2 = Generator3DLUT_zero()
            self.LUT3 = Generator3DLUT_zero()
            self.LUT4 = Generator3DLUT_zero()
            self.LUT5 = Generator3DLUT_zero()
        elif self.n_lut==8:
            self.LUT0 = Generator3DLUT_identity()
            self.LUT1 = Generator3DLUT_zero()
            self.LUT2 = Generator3DLUT_zero()
            self.LUT3 = Generator3DLUT_zero()
            self.LUT4 = Generator3DLUT_zero()
            self.LUT5 = Generator3DLUT_zero()
            self.LUT6 = Generator3DLUT_zero()
            self.LUT7 = Generator3DLUT_zero()
        # else:
        #     self.lut_list=[]
        #     self.lut_list=self.initial_luts(self.n_lut)

        self.backbone=backbone
        if self.backbone=='idih':
            self.classifier = Weight_predictor_idih(in_channels, n_lut)
        else:
            self.classifier = Weight_predictor_issam(in_channels, n_lut, self.fb)

    # def initial_luts(self, lut_num=3):
    #     self.lut_list.append(Generator3DLUT_identity())
    #     for ii in range(lut_num-1):
    #         self.lut_list.append(Generator3DLUT_zero())
    #     return self.lut_list

    def forward(self, encoder_outputs, image, mask):
        pred_weights = self.classifier(encoder_outputs, mask)
        if len(pred_weights.shape) == 1:
            pred_weights = pred_weights.unsqueeze(0)
        combine_A = image.new(image.size())
        if self.n_lut==3:
            gen_A0 = self.LUT0(image)
            gen_A1 = self.LUT1(image)
            gen_A2 = self.LUT2(image)
            for b in range(image.size(0)):
                combine_A[b,:,:,:] = pred_weights[b,0] * gen_A0[b,:,:,:] + pred_weights[b,1] * gen_A1[b,:,:,:] + pred_weights[b,2] * gen_A2[b,:,:,:] #+ pred[b,3] * gen_A3[b,:,:,:] + pred[b,4] * gen_A4[b,:,:,:]
        elif self.n_lut==2:
            gen_A0 = self.LUT0(image)
            gen_A1 = self.LUT1(image)
            # combine_A = gen_A1
            for b in range(image.size(0)):
                combine_A[b,:,:,:] = pred_weights[b,0] * gen_A0[b,:,:,:] + pred_weights[b,1] * gen_A1[b,:,:,:]
        elif self.n_lut==1:
            gen_A0 = self.LUT0(image)
            # combine_A = gen_A0
            for b in range(image.size(0)):
                combine_A[b,:,:,:] = pred_weights[b,0] * gen_A0[b,:,:,:]
        elif self.n_lut==4:
            gen_A0 = self.LUT0(image)
            gen_A1 = self.LUT1(image)
            gen_A2 = self.LUT2(image)
            gen_A3 = self.LUT3(image)
            for b in range(image.size(0)):
                combine_A[b,:,:,:] = pred_weights[b,0] * gen_A0[b,:,:,:] + pred_weights[b,1] * gen_A1[b,:,:,:] + pred_weights[b,2] * gen_A2[b,:,:,:] \
                    + pred_weights[b,3] * gen_A3[b,:,:,:] #+ pred_weights[b,4] * gen_A4[b,:,:,:]
        elif self.n_lut==5:
            gen_A0 = self.LUT0(image)
            gen_A1 = self.LUT1(image)
            gen_A2 = self.LUT2(image)
            gen_A3 = self.LUT3(image)
            gen_A4 = self.LUT4(image)
            for b in range(image.size(0)):
                combine_A[b,:,:,:] = pred_weights[b,0] * gen_A0[b,:,:,:] + pred_weights[b,1] * gen_A1[b,:,:,:] + pred_weights[b,2] * gen_A2[b,:,:,:] \
                    + pred_weights[b,3] * gen_A3[b,:,:,:] + pred_weights[b,4] * gen_A4[b,:,:,:]
        elif self.n_lut==6:
            gen_A0 = self.LUT0(image)
            gen_A1 = self.LUT1(image)
            gen_A2 = self.LUT2(image)
            gen_A3 = self.LUT3(image)
            gen_A4 = self.LUT4(image)
            gen_A5 = self.LUT5(image)
            for b in range(image.size(0)):
                combine_A[b,:,:,:] = pred_weights[b,0] * gen_A0[b,:,:,:] + pred_weights[b,1] * gen_A1[b,:,:,:] + pred_weights[b,2] * gen_A2[b,:,:,:] \
                    + pred_weights[b,3] * gen_A3[b,:,:,:] + pred_weights[b,4] * gen_A4[b,:,:,:] + pred_weights[b,5] * gen_A5[b,:,:,:]
        elif self.n_lut==8:
            gen_A0 = self.LUT0(image)
            gen_A1 = self.LUT1(image)
            gen_A2 = self.LUT2(image)
            gen_A3 = self.LUT3(image)
            gen_A4 = self.LUT4(image)
            gen_A5 = self.LUT5(image)
            gen_A6 = self.LUT6(image)
            gen_A7 = self.LUT7(image)
            for b in range(image.size(0)):
                combine_A[b,:,:,:] = pred_weights[b,0] * gen_A0[b,:,:,:] + pred_weights[b,1] * gen_A1[b,:,:,:] + pred_weights[b,2] * gen_A2[b,:,:,:] \
                    + pred_weights[b,3] * gen_A3[b,:,:,:] + pred_weights[b,4] * gen_A4[b,:,:,:] + pred_weights[b,5] * gen_A5[b,:,:,:] \
                    + pred_weights[b,6] * gen_A6[b,:,:,:] + pred_weights[b,7] * gen_A7[b,:,:,:]
        # add the clamping
        if self.clamp:
            combine_A = torch.clamp(combine_A,0,1)

        combine_A = combine_A*mask+image*(1-mask)
        
        return combine_A#,pred_weights.squeeze().detach().cpu().numpy()


class Generator3DLUT_identity(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_identity, self).__init__()
        if dim == 33:
            file = open("IdentityLUT33.txt", 'r')
        elif dim == 64:
            file = open("IdentityLUT64.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((3,dim,dim,dim), dtype=np.float32)

        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    n = i * dim*dim + j * dim + k
                    x = lines[n].split()
                    buffer[0,i,j,k] = float(x[0])
                    buffer[1,i,j,k] = float(x[1])
                    buffer[2,i,j,k] = float(x[2])
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)
        #self.LUT, output = self.TrilinearInterpolation(self.LUT, x)
        return output

class Generator3DLUT_zero(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_zero, self).__init__()

        self.LUT = torch.zeros(3,dim,dim,dim, dtype=torch.float)
        self.LUT = nn.Parameter(self.LUT.clone().detach().requires_grad_(True))
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):

        _, output = self.TrilinearInterpolation(self.LUT, x)
        return output

class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()

        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim-1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)
        
        assert 1 == trilinear.forward(lut, 
                                      x, 
                                      output,
                                      dim, 
                                      shift, 
                                      binsize, 
                                      W, 
                                      H, 
                                      batch)

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]
        
        ctx.save_for_backward(*variables)
        
        return lut, output
    
    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])
            
        assert 1 == trilinear.backward(x, 
                                       x_grad, 
                                       lut_grad,
                                       dim, 
                                       shift, 
                                       binsize, 
                                       W, 
                                       H, 
                                       batch)
        return lut_grad, x_grad


class TrilinearInterpolation(torch.nn.Module):
    def __init__(self):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        return TrilinearInterpolationFunction.apply(lut, x)


class TrilinearInterpolationGS(torch.nn.Module):
    def __init__(self):
        super(TrilinearInterpolationGS, self).__init__()

    def forward(self, lut, img):
        # scale im between -1 and 1 since its used as grid input in grid_sample
        img = (img - .5) * 2.
        # grid_sample expects NxD_outxH_outxW_outx3 (1x1xHxWx3)
        img = img.permute(0, 2, 3, 1)[:, None]
        # add batch dim to LUT
        lut = lut[None] # [B,C,D_in,H_in,W_in] -> [B,3,M,M,M] 
        # grid sample
        result = F.grid_sample(lut, img, mode='bilinear', padding_mode='border', align_corners=True) # [B, C, D_out, H_out, W_out ]
        # drop added dimensions and permute back
        result = result[:, :, 0,:,:]
        # print('after result', result.shape)
        return lut,result