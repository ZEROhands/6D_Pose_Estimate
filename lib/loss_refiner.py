from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
from lib.knn.__init__ import KNearestNeighbor


def loss_calculation(pred_r, pred_t, target, model_points, idx, points, num_point_mesh, sym_list):
    # knn = KNearestNeighbor(1)
    # print(pred_r.size())
    bs, num_p, _ = pred_r.size()
    # print('bs',bs)

    # pred_r = pred_r.view(1, 1, -1)
    # pred_t = pred_t.view(1, 1, -1)
    # bs, num_p, _ = pred_r.size()
    num_input_points = len(points[0])

    pred_r = pred_r / (torch.norm(pred_r, dim=2)).view(bs, num_p, 1)

    base = torch.cat(((1.0 - 2.0 * (pred_r[:, :, 2] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] - 2.0 * pred_r[:, :, 0] * pred_r[:, :, 3]).view(bs,
                                                                                                               num_p,
                                                                                                               1), \
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs,
                                                                                                               num_p,
                                                                                                               1), \
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 3] * pred_r[:, :, 0]).view(bs,
                                                                                                               num_p,
                                                                                                               1), \
                      (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs,
                                                                                                                num_p,
                                                                                                                1), \
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs,
                                                                                                                num_p,
                                                                                                                1), \
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs,
                                                                                                               num_p,
                                                                                                               1), \
                      (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 2] ** 2)).view(bs, num_p, 1)),
                     dim=2).contiguous().view(bs * num_p, 3, 3)

    ori_base = base
    base = base.contiguous().transpose(2, 1).contiguous()
    # print("bs",bs)
    # print(model_points.shape)
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh,
                                                                                           3)
    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    ori_target = target
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t
    
    pred = torch.add(torch.bmm(model_points, base), pred_t)
    for i in range(bs):
        if idx[i].item() in sym_list:
            target1 = target[num_p*i].transpose(1, 0).contiguous().view(3, -1)
            pred1 = pred[num_p*i:num_p*(i+1),:,:].permute(2, 0, 1).contiguous().view(3, -1)
            # print("target shape是{0}\npred shape是{1}".format(target.unsqueeze(0).shape,pred.unsqueeze(0)[0][0]))
            inds = KNearestNeighbor.apply(target1.unsqueeze(0), pred1.unsqueeze(0))
            target1 = torch.index_select(target1, 1, inds.view(-1).detach() - 1)
            # print("target shape是{}".format(target.shape))
            target[num_p*i:num_p*(i+1),:,:] = target1.view(3, num_p, num_point_mesh).permute(1, 2, 0).contiguous()
            pred[num_p*i:num_p*(i+1),:,:] = pred1.view(3, num_p, num_point_mesh).permute(1, 2, 0).contiguous()

    # print(torch.norm((pred - target), dim=2),'and shape',torch.norm((pred - target), dim=2).shape)
    dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)

    # print(dis.shape)
    # print(dis)
    # print("ori_base.shape:",ori_base.shape)
    # print("ori_t.shape:",ori_t.shape)

    points1 = points.view(bs, num_input_points, 3).cuda()
    new_points = torch.zeros([bs, num_input_points, 3]).cuda()
    new_target = torch.zeros([bs, num_point_mesh, 3]).cuda()
    for i in range(bs):
        t = ori_t[i]
        points_tmp = points1[i].view(1, num_input_points, 3)
        ori_base_1 = ori_base[i].view(1, 3, 3).contiguous()
        ori_t_1 = t.repeat(num_input_points, 1).contiguous().view(1, num_input_points, 3)
        new_points[i] = torch.bmm((points_tmp - ori_t_1), ori_base_1).contiguous()
        new_target[i] = ori_target[i].view(1, num_point_mesh, 3).contiguous()
        ori_t_1 = t.repeat(num_point_mesh, 1).contiguous().view(1, num_point_mesh, 3)
        new_target[i] = torch.bmm((new_target[i] - ori_t_1), ori_base_1).contiguous()

    # print('------------> ', dis.item(), idx[0].item())
    # del knn
    return dis, new_points.detach(), new_target.detach()


class Loss_refine(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss_refine, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, target, model_points, idx, points):
        return loss_calculation(pred_r, pred_t, target, model_points, idx, points, self.num_pt_mesh, self.sym_list)

