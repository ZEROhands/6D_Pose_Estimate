from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
from lib.knn.__init__ import KNearestNeighbor


def loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, num_point_mesh, sym_list):
    # print(pred_r.shape)
    # knn = KNearestNeighbor(1)
    bs, num_p, _ = pred_c.size()

    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))
    
    base = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),\
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)

    ori_base = base
    base = base.contiguous().transpose(2, 1).contiguous()
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    ori_target = target
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t
    points = points.contiguous().view(bs * num_p, 1, 3)
    pred_c = pred_c.contiguous().view(bs * num_p)
    # base,points,pred_t =     Variable(base).cuda(), \
    #                         Variable(points).cuda(), \
    #                         Variable(pred_t).cuda()
    pred = torch.add(torch.bmm(model_points, base), points + pred_t)

    if not refine:
        for i in range(bs):
            if idx[i].item() in sym_list:
                target1 = target[num_p*i].transpose(1, 0).contiguous().view(3, -1)

                pred1 = pred[num_p*i:num_p*(i+1),:,:].permute(2, 0, 1).contiguous().view(3, -1)
                # inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
                inds = KNearestNeighbor.apply(target1.unsqueeze(0), pred1.unsqueeze(0))
                target1 = torch.index_select(target1, 1, inds.view(-1).detach() - 1)
                target[num_p*i:num_p*(i+1),:,:] = target1.view(3,  num_p, num_point_mesh).permute(1, 2, 0).contiguous()
                pred[num_p*i:num_p*(i+1),:,:] = pred1.view(3, num_p, num_point_mesh).permute(1, 2, 0).contiguous()

    dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)
    # print(dis.shape)
    loss = torch.norm((dis * pred_c - w * torch.log(pred_c)), dim=0)
    

    pred_c = pred_c.view(bs, num_p)
    how_max, which_max = torch.max(pred_c, 1)
    # print("which_max:",which_max)
    dis = dis.view(bs, num_p)

    # print('ori_t:', ori_t.shape)
    # print('points:',points.shape)
    # print('ori_base:', ori_base.shape)
    # print('ori_target',ori_target.shape)
    points_1 = points.view(bs, num_p, 3).cuda()
    new_points = torch.zeros([bs, num_p, 3]).cuda()
    new_target = torch.zeros([bs, num_point_mesh, 3]).cuda()
    # print('bs',bs)
    # print('num_p',num_p)
    # print('num_mesh',num_point_mesh)
    # print('ori_tar',ori_target.shape)
    for i in range(bs):
        t = ori_t[which_max[i]+num_p*i] + points[which_max[i]+num_p*i]  #[bs,3]
        tmp_points = points_1[i].view(1, num_p, 3)
        ori_base1 = ori_base[which_max[i]+num_p*i].view(1, 3, 3).contiguous()
        ori_t_1 = t.repeat(num_p, 1).contiguous().view(1, num_p, 3)
        new_points[i] = torch.bmm((tmp_points - ori_t_1), ori_base1).contiguous()
        # print('look', ori_target[num_p * i].shape)
        new_target[i] = ori_target[num_p*i].view(1, num_point_mesh, 3).contiguous()

        ori_t2 = t.repeat(num_point_mesh, 1).contiguous().view(1, num_point_mesh, 3)
        new_target[i] = torch.bmm((new_target[i] - ori_t2), ori_base1).contiguous()





    # print('------------> ', dis[0][which_max[0]].item(), pred_c[0][which_max[0]].item(), idx[0].item())
    # del knn
    dis_end = torch.zeros(bs,1)
    for i in range(bs):
        dis_end[i] = dis[i][which_max[i]]
    # print("new_points:",new_points.shape)
    return loss, dis_end, new_points.detach(), new_target.detach()


class Loss(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine):

        return loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, self.num_pt_mesh, self.sym_list)

