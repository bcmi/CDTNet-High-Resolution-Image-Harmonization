import torch
import torch.nn as nn

from iharm.utils import misc


class Loss(nn.Module):
    def __init__(self, pred_outputs, gt_outputs):
        super().__init__()
        self.pred_outputs = pred_outputs
        self.gt_outputs = gt_outputs


class MSE(Loss):
    def __init__(self, pred_name='images', gt_image_name='target_images'):
        super(MSE, self).__init__(pred_outputs=(pred_name,), gt_outputs=(gt_image_name,))

    def forward(self, pred, label):
        label = label.view(pred.size())
        loss = torch.mean((pred - label) ** 2, dim=misc.get_dims_with_exclusion(label.dim(), 0))
        return loss


class MaskWeight_MSE(Loss):
    def __init__(self, min_area=100, pred_name='images',
                 gt_image_name='target_images', gt_mask_name='masks'):
        super(MaskWeight_MSE, self).__init__(pred_outputs=(pred_name, ),
                                              gt_outputs=(gt_image_name, gt_mask_name))
        self.min_area = min_area

    def forward(self, pred, label, mask):
        label = label.view(pred.size())
        reduce_dims = misc.get_dims_with_exclusion(label.dim(), 0)

        loss = (pred - label) ** 2
        delimeter = pred.size(1) * torch.clamp_min(torch.sum(mask, dim=reduce_dims), self.min_area)
        loss = torch.sum(loss, dim=reduce_dims) / delimeter

        return loss

class TV_3D(nn.Module):
    def __init__(self, dim=33):
        super(TV_3D,self).__init__()

        self.weight_r = torch.ones(3,dim,dim,dim-1, dtype=torch.float)
        self.weight_r[:,:,:,(0,dim-2)] *= 2.0
        self.weight_g = torch.ones(3,dim,dim-1,dim, dtype=torch.float)
        self.weight_g[:,:,(0,dim-2),:] *= 2.0
        self.weight_b = torch.ones(3,dim-1,dim,dim, dtype=torch.float)
        self.weight_b[:,(0,dim-2),:,:] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT_list):
        # tv_all = 0
        mn_all = 0
        for LUT in LUT_list:
            dif_r = LUT.LUT[:,:,:,:-1] - LUT.LUT[:,:,:,1:]
            dif_g = LUT.LUT[:,:,:-1,:] - LUT.LUT[:,:,1:,:]
            dif_b = LUT.LUT[:,:-1,:,:] - LUT.LUT[:,1:,:,:]
            # tv = torch.mean(torch.mul((dif_r ** 2),self.weight_r)) + torch.mean(torch.mul((dif_g ** 2),self.weight_g)) + torch.mean(torch.mul((dif_b ** 2),self.weight_b))
            mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))
            # tv_all += tv
            mn_all += mn
        # return tv_all, mn_all
        return mn_all