import argparse
import os
import random
import math
import copy
import torch
from functools import reduce
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet
import lib.graph_conv as gc
from lib.parameters import *
from lib.model import  *


# if torch.cuda.is_available():
#     torch_device = torch.device('cuda')

# para = Parameters()
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
def zhijiao(input, R):
    R = R
    Theta = input[:, :, 1] * np.pi
    Phi = input[:, :, 2] * 2 * np.pi - np.pi
    # print(np.sin(Theta))
    x = R * torch.sin(Theta) * torch.cos(Phi)
    # print(x.shape)
    y = R * torch.sin(Theta) * torch.sin(Phi)
    z = R * torch.cos(Theta)
    out = torch.stack([x, y, z], axis=2)
    return out


psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]().cuda()
        # self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


def zhijiao(input, R):
    R = R
    Theta = input[:, :, 1] * np.pi
    Phi = input[:, :, 2] * 2 * np.pi - np.pi
    x = R * torch.sin(Theta) * torch.cos(Phi)
    y = R * torch.sin(Theta) * torch.sin(Phi)
    z = R * torch.cos(Theta)
    out = torch.stack([x, y, z], axis=2)
    return out



class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        '''
        :param in_channels:  ??????????????????
        :param out_channels: ??????????????????   ???????????? ??????????????????????????????
        :param stride:  ??????????????????1
        :param M:  ?????????
        :param r: ??????Z???????????????????????????d ?????????????????????????????? ??????S->Z ??????????????????????????? ??????????????????
        :param L:  ?????????????????????Z?????????????????????32
        '''
        super(SKConv,self).__init__()
        d=max(in_channels//r,L)   # ????????????Z ?????????d
        self.M=M
        self.out_channels=out_channels
        # self.conv=nn.ModuleList()  # ?????????????????? ?????? ????????????????????????
        # for i in range(M):
        #     # ?????????????????????????????? ????????????5x5??? ???3X3???dilation=2??????????????? ???????????????????????????G=32
        #     self.conv.append(nn.Sequential(nn.Conv1d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=32,bias=False),
        #                                    nn.BatchNorm1d(out_channels),
        #                                    nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool1d(1) # ?????????pool???????????????    ???????????????1????????? GAP
        self.fc1=nn.Sequential(nn.Conv1d(out_channels,d,1,bias=False),
                               nn.BatchNorm1d(d),
                               nn.ReLU(inplace=True))   # ??????
        self.fc2=nn.Conv1d(d,out_channels*M,1,1,bias=False)  # ??????
        self.softmax=nn.Softmax(dim=1) # ??????dim=1  ??????????????????????????????????????????softmax,?????? ????????????a+b+..=1
    def forward(self, input1, input2):
        batch_size=input1.size(0)
        output=[]
        #the part of split
        # for i,conv in enumerate(self.conv):
            #print(i,conv(input).size())
        output.append(input1)
        output.append(input2)
        #the part of fusion
        U=reduce(lambda x,y:x+y,output) # ????????????????????? ????????????U
        s=self.global_pool(U)
        z=self.fc1(s)  # S->Z??????
        a_b=self.fc2(z) # Z->a???b ??????  ????????????conv 1x1????????????????????????????????????????????????a,????????????b
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) #????????????????????? ????????????????????????
        a_b=self.softmax(a_b) # ??????????????????????????????????????????softmax
        #the part of selection
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b   chunk???pytorch????????????tensor??????????????????????????? ??????tensor???
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1),a_b)) # ???????????????  ??????????????????????????????
        # print('output:',output[0].shape)
        # print('a_b:',a_b[0].shape)
        V=list(map(lambda x,y:x*y,output,a_b)) # ???????????????  ????????????????????????U ???????????????
        V=reduce(lambda x,y:x+y,V) # ???????????????????????? ???????????????
        return V



class SEblock(nn.Module):  # ?????????????????????
    def __init__(self, channel, r=0.5):  # channel??????????????????, r???????????????????????????->?????????????????????
        super(SEblock, self).__init__()
        # ??????????????????
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # ????????????
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel * r)),  # int(channel * r)?????????
            nn.ReLU(),
            nn.Linear(int(channel * r), channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # ???x????????????????????????, ????????????????????????
        branch = self.global_avg_pool(x)
        branch = branch.view(branch.size(0), -1)


        # ????????????????????????
        weight = self.fc(branch)
        # print(weight.shape)

        # ????????????b, c???weight, reshape???b, c, 1, 1 ??? ??????x ??????
        b,c = weight.shape
        weight = torch.reshape(weight, (b, c, 1))

        # ??????????????????
        scale = weight * x
        return scale

class AFP(nn.Module):
    def __init__(self,channel):
        super(AFP, self).__init__()

        self.branch1 = nn.Sequential(
            nn.MaxPool1d(3, 1, padding=1),  # 1.??????????????????,??????????????????????????????3, ?????????stride??????padding, ??????????????????????????????, ??????(3, 1, 1)
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool1d(3, 1, padding=1),  # 2.??????????????????, ??????????????????????????????2, ?????????stride??????padding, ??????????????????????????????, ??????(3, 1, 1)
        )

        self.branch3_1 = nn.Sequential(
            nn.Conv1d(channel, int(channel/4), 1),
            nn.Conv1d(int(channel/4), int(channel/4), 3, padding=1),  # 3_1??????, ??????1??1????????????????????????, ??????????????????3??3????????????????????????, ??????????????????3//2, ????????????????????????1
            nn.Conv1d(int(channel/4), int(channel/4), 3, padding=1),
        )

        self.branch3_2 = nn.Sequential(
            nn.Conv1d(channel, int(channel*3/4), 1),  # 3_2??????, ??????1??1??????????????????????????????, ?????????????????????3, ????????????1, ??????????????????????????????2
            nn.Conv1d(int(channel*3/4), int(channel*3/4), 3, padding=1)
        )
        # ???????????????
        self.branch_SE = SEblock(channel = channel)

        # ??????????????????????????????
        # nn.Parameter ??????????????????, ??????????????????????????????, ???????????????????????????????????????????????????, ?????????????????????????????????Parameter??????, ?????????????????????????????????????????????

        # self.w = nn.Parameter(torch.ones(5))  # 4?????????, ?????????????????????????????????????????????, ????????????1, nn.Parameter?????????Tensor???????????????
        # self.w = nn.Parameter(torch.Tensor([0.5, 0.25, 0.15, 0.1]), requires_grad=False)  # ???????????????????????????, ???????????????, ???????????????

    def forward(self, x,weight):
        b1 = self.branch1(x)
        b2 = self.branch2(x)

        b3_1 = self.branch3_1(x)
        b3_2 = self.branch3_2(x)
        b3_Combine = torch.cat((b3_1, b3_2), dim=1)
        b3 = self.branch_SE(b3_Combine)

        b4 = x
        #
        # print("b1:", b1.shape)
        # print("b2:", b2.shape)
        # print("b3:", b3.shape)
        # print("b4:", b4.shape)

        # ???????????????
        w1 = torch.exp(weight[0]) / torch.sum(torch.exp(weight))
        w2 = torch.exp(weight[1]) / torch.sum(torch.exp(weight))
        w3 = torch.exp(weight[2]) / torch.sum(torch.exp(weight))
        w4 = torch.exp(weight[3]) / torch.sum(torch.exp(weight))

        # ???????????????
        x_out = b1 * w1 + b2 * w2 + b3 * w3 + b4 * w4
        # print("??????????????????:", x_out.shape)
        return x_out


class PoseNetFeat(nn.Module):
    def __init__(self, num_points, batch_size):
        super(PoseNetFeat, self).__init__()
        self.fusion1 = AFP(32)
        self.fusion2 = AFP(128)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.g_conv1 = torch.nn.Conv1d(128, 64, 1)
        self.g_conv2 = torch.nn.Conv1d(256, 128, 1)

        # self.conv5 = torch.nn.Conv1d(256, 512, 1)
        # self.conv6 = torch.nn.Conv1d(512, 1024, 1)
        self.conv5 = torch.nn.Conv1d(128, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 512, 1)

        self.fusion_conv1 = SKConv(64, 64)
        self.fusion_conv2 = SKConv(128, 128)

        self.tarcon = torch.nn.Conv1d(3, 64, 1)
        self.ap1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points

        self.lay1 = torch.nn.BatchNorm1d(64)
        self.lay2 = torch.nn.BatchNorm1d(64)
        self.lay3 = torch.nn.BatchNorm1d(128)
        self.lay4 = torch.nn.BatchNorm1d(128)
        # self.lay1 = torch.nn.LayerNorm(64)
        # self.lay2 = torch.nn.LayerNorm(64)
        # self.lay3 = torch.nn.LayerNorm(128)
        # self.lay4 = torch.nn.LayerNorm(128)

        # self.enc1 = t.Encoder(t.EncoderLayer(size=(batch_size, 500, 64),
        #                                      self_attn=t.MultiHeadedAttention(8, 64, 0.4),
        #                                      feed_forward=t.PositionwiseFeedForward(64, 128, 0.4),
        #                                      dropout=0.4), 1)


    def forward(self, emb, g1, g2, weight1, weight2, weight3, weight4):

        emb = self.fusion1(emb, weight1)

        # emb = F.relu(self.lay1(self.e_conv1(emb).transpose(2,1)).transpose(2,1))
        emb = F.relu(self.lay1(self.e_conv1(emb)))


        g1 = self.fusion2(g1, weight2)

        # g = F.relu(self.lay2(self.g_conv1(g1).transpose(2,1)).transpose(2,1))  #[1,64,500]
        g = F.relu(self.lay2(self.g_conv1(g1)))  #[1,64,500]
        # print(g)

        # pointfeat_1 = torch.cat((emb, g), dim=1)
        # w1 = (torch.exp(weight3[0]) ) / (torch.sum(torch.exp(weight3)) )
        # w2 = torch.exp(weight3[1]) / (torch.sum(torch.exp(weight3)) )
        # pointfeat_1 = emb * w1 + g * w2  # 64
        # pointfeat_1 = self.enc1(pointfeat_1.transpose(2, 1), None).transpose(2, 1) #64
        pointfeat_1 = self.fusion_conv1(emb,g)
        emb = F.relu(self.lay3(self.e_conv2(emb)))
        g = F.relu(self.lay4(self.g_conv2(g2)))

        # pointfeat_2 = torch.cat((emb, g), dim=1)
        # w3 = (torch.exp(weight4[0])) / (torch.sum(torch.exp(weight4)) )
        # w4 = torch.exp(weight4[1]) / (torch.sum(torch.exp(weight4)) )
        # pointfeat_2 = emb * w3 + g * w4 #64
        pointfeat_2 = self.fusion_conv2(emb,g)


        # print("pointfeat_2:{}".format(pointfeat_2))

        x = F.relu(self.conv5(pointfeat_2))
        # print("x{}".format(x))
        x = F.relu(self.conv6(x))

        # x = self.bn3(x)
        ap_x = self.ap1(x)

        # ap_x = torch.cat([ap_x1, ap_x2, ap_x], 1) #128+256+1024=1408

        ap_x = ap_x.view(-1, 512, 1).repeat(1, 1, self.num_points)
        # ap_x = ap_x.view(-1, 1408, 1).repeat(1, 1, self.num_points)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1)  # 128 + 128 + 512
        # return ap_x
        # return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1)  # 128 + 256 + 1408=1792


class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj, para, batch_size):
        # def __init__(self, num_points, num_obj):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.para = para
        self.graph_net = DGCNN_cls(1024, 10, 0.5, 500)
        self.cnn = ModifiedResnet()
        self.w = nn.Parameter(torch.ones(4))
        self.w1 = nn.Parameter(torch.ones(4))
        self.w2 = nn.Parameter(torch.ones(2))
        self.w3 = nn.Parameter(torch.ones(2))
        self.feat = PoseNetFeat(num_points, batch_size)

        self.conv1_r = torch.nn.Conv1d(704, 640, 1)
        self.conv1_t = torch.nn.Conv1d(704, 640, 1)
        self.conv1_c = torch.nn.Conv1d(704, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj * 4, 1)  # quaternion
        self.conv4_t = torch.nn.Conv1d(128, num_obj * 3, 1)  # translation
        self.conv4_c = torch.nn.Conv1d(128, num_obj * 1, 1)  # confidence

        self.num_obj = num_obj

        # self.feat1_attention = Cbam(1408)
        # self.feat2_attention = Cbam(640)

    def forward(self, img, x, R, choose, obj):
        out_img = self.cnn(img)

        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1).long()
        emb = torch.gather(emb, 2, choose).contiguous()

        x1 = x.transpose(2, 1).contiguous()
        x1,x2,_ = self.graph_net(x1)
        # x1 = x1.transpose(2, 1).contiguous()



        ap_x = self.feat(emb, x1, x2, self.w, self.w1, self.w2, self.w3)
        #print('RGB????????????:{0:.4f},{1:.4f},{2:.4f},{3:.4f}'.format(self.w[0], self.w[1], self.w[2], self.w[3]))
        #print('Graph????????????:{0:.4f},{1:.4f},{2:.4f},{3:.4f}'.format(self.w1[0], self.w1[1], self.w1[2], self.w1[3]))
        #print('RGB??????:{0:.4f}-point??????:{1:.4f}'.format(self.w2[0], self.w2[1]))
        #print('RGB??????:{0:.4f}-point??????:{1:.4f}'.format(self.w3[0], self.w3[1]))
        # print('apx.shape:{0}'.format(ap_x.shape))
        # ap_x = self.feat1_attention(ap_x)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))

        # rx = self.feat2_attention(rx)
        # tx = self.feat2_attention(tx)
        # cx = self.feat2_attention(cx)

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)

        out_rx = torch.empty(bs, 4, self.num_points).cuda()
        out_tx = torch.empty(bs, 3, self.num_points).cuda()
        out_cx = torch.empty(bs, 1, self.num_points).cuda()

        for i in range(bs):
            b = i
            out_rx[i] = torch.index_select(rx[b], 0, obj[b])
            out_tx[i] = torch.index_select(tx[b], 0, obj[b])
            out_cx[i] = torch.index_select(cx[b], 0, obj[b])

        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()

        return out_rx, out_tx, out_cx, emb.detach()


class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points, batch_size):
        super(PoseRefineNetFeat, self).__init__()
        self.fusion1 = AFP(32)
        self.fusion2 = AFP(128)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.g_conv1 = torch.nn.Conv1d(128, 64, 1)
        self.g_conv2 = torch.nn.Conv1d(256, 128, 1)

        # self.conv5 = torch.nn.Conv1d(256, 512, 1)
        # self.conv6 = torch.nn.Conv1d(512, 1024, 1)
        self.conv5 = torch.nn.Conv1d(192, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 512, 1)

        self.fusion_conv1 = SKConv(64, 64)
        self.fusion_conv2 = SKConv(128, 128)

        self.tarcon = torch.nn.Conv1d(3, 64, 1)
        self.ap1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points

        self.lay1 = torch.nn.BatchNorm1d(64)
        self.lay2 = torch.nn.BatchNorm1d(64)
        self.lay3 = torch.nn.BatchNorm1d(128)
        self.lay4 = torch.nn.BatchNorm1d(128)
        # self.lay1 = torch.nn.LayerNorm(64)
        # self.lay2 = torch.nn.LayerNorm(64)
        # self.lay3 = torch.nn.LayerNorm(128)
        # self.lay4 = torch.nn.LayerNorm(128)
        # self.enc1 = t.Encoder(t.EncoderLayer(size=(batch_size, 500, 64),
        #                                      self_attn=t.MultiHeadedAttention(8, 64, 0.4),
        #                                      feed_forward=t.PositionwiseFeedForward(64, 128, 0.4),
        #                                      dropout=0.4), 1)

    def forward(self, emb, g1, g2, weight1, weight2, weight3, weight4):
        emb = self.fusion1(emb, weight1)
        emb = F.relu(self.lay1(self.e_conv1(emb)))

        g1 = self.fusion2(g1, weight2)
        g = F.relu(self.lay2(self.g_conv1(g1)))  # [1,64,500]

        # pointfeat_1 = torch.cat((emb, g), dim=1)
        # w1 = (torch.exp(weight3[0])) / (torch.sum(torch.exp(weight3)))
        # w2 = torch.exp(weight3[1]) / (torch.sum(torch.exp(weight3)))
        # pointfeat_1 = emb * w1 + g * w2  # 64
        # pointfeat_1 = self.enc1(pointfeat_1.transpose(2, 1), None).transpose(2, 1)  # 128
        pointfeat_1 = self.fusion_conv1(emb,g)
        emb = F.relu(self.lay3(self.e_conv2(emb)))
        g = F.relu(self.lay4(self.g_conv2(g2)))

        # pointfeat_2 = torch.cat((emb, g), dim=1)
        # w3 = (torch.exp(weight4[0])) / (torch.sum(torch.exp(weight4)))
        # w4 = torch.exp(weight4[1]) / (torch.sum(torch.exp(weight4)))
        # pointfeat_2 = emb * w3 + g * w4
        pointfeat_2 = self.fusion_conv2(emb,g)
        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)


        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        # x = self.bn3(x)
        ap_x = self.ap1(x)

        # ap_x = torch.cat([ap_x1, ap_x2, ap_x], 1) #128+256+1024=1408

        ap_x = ap_x.view(-1, 512)
        # print("ap_x",ap_x.shape)
        return ap_x


class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj, para, batch_size):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat1 = PoseRefineNetFeat(num_points, batch_size)
        self.graph_net = DGCNN_cls(1024, 10, 0.5, 500)
        self.w_r = nn.Parameter(torch.ones(4))
        self.w_r1 = nn.Parameter(torch.ones(4))
        self.w_r2 = nn.Parameter(torch.ones(2))
        self.w_r3 = nn.Parameter(torch.ones(2))


        self.conv1_r = torch.nn.Linear(512, 512)
        self.conv1_t = torch.nn.Linear(512, 512)
        self.conv1_r_dr = torch.nn.Dropout(0.5)
        self.conv1_t_dr = torch.nn.Dropout(0.5)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)
        # self.conv2_r_dr = torch.nn.Dropout(0.5)
        # self.conv2_t_dr = torch.nn.Dropout(0.5)

        self.conv3_r = torch.nn.Linear(128, num_obj * 4)  # quaternion
        self.conv3_t = torch.nn.Linear(128, num_obj * 3)  # translation
        # self.conv3_r_dr = torch.nn.Dropout(0.5)
        # self.conv3_t_dr = torch.nn.Dropout(0.5)
        self.num_obj = num_obj
        # self.feat1_attention = Cbam(1024)
        # self.feat2_attention = Cbam(512)

    def forward(self, emb, x, R, obj):
        bs = x.size()[0]
        x1 = x.transpose(2, 1).contiguous()
        x1,x2,_ = self.graph_net(x1)
        # x1 = x1.transpose(2, 1).contiguous()

        ap_x = self.feat1(emb, x1, x2, self.w_r, self.w_r1, self.w_r2, self.w_r3)
        # print('RGB????????????:{0:.4f},{1:.4f},{2:.4f},{3:.4f}'.format(self.w_r[0], self.w_r[1], self.w_r[2], self.w_r[3]))
        # print('Graph????????????:{0:.4f},{1:.4f},{2:.4f},{3:.4f}'.format(self.w_r1[0], self.w_r1[1], self.w_r1[2], self.w_r1[3]))
        # print('RGB??????:{0:.4f}-point??????:{1:.4f}'.format(self.w_r2[0], self.w_r2[1]))
        # print('RGB??????:{0:.4f}-point??????:{1:.4f}'.format(self.w_r3[0], self.w_r3[1]))

        # ap_x = self.feat1_attention(ap_x)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        # print('rx',rx.shape)
        rx = self.conv1_r_dr(rx)
        tx = self.conv1_t_dr(tx)
        # rx = self.feat1_attention(rx)
        # tx = self.feat1_attention(tx)

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        # rx = self.conv2_r_dr(rx)
        # tx = self.conv2_t_dr(tx)

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)
        # rx = self.conv3_r_dr(rx)
        # tx = self.conv3_t_dr(tx)
        out_rx = torch.empty(bs,1, 4).cuda()
        out_tx = torch.empty(bs,1, 3).cuda()
        for i in range(bs):
            b = i
            out_rx[i] = torch.index_select(rx[b], 0, obj[b])
            out_tx[i] = torch.index_select(tx[b], 0, obj[b])
        # print("out_rx",out_rx.shape)
        return out_rx, out_tx















if __name__ == '__main__':
    # net = ModifiedResnet()
    objlist = 13
    num_points = 500
    # net = PoseNetFeat(500)
    net = PoseNet(num_points, objlist)
    # a,b,c = PoseNet(num_points,objlist)
    # net = PoseRefineNetFeat(500)
    # net = PoseRefineNet(num_points,objlist)

    print(net)
