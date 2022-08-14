
import torch
import torch.nn as nn
import torch.nn.parallel

import torch.utils.data

import torch.nn.functional as F
from lib.pspnet import PSPNet


from lib.model import  *



# if torch.cuda.is_available():
#     torch_device = torch.device('cuda')

# para = Parameters()

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

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x








class SEblock(nn.Module):  # 注意力机制模块
    def __init__(self, channel, r=0.5):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(SEblock, self).__init__()
        # 全局均值池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel * r)),  # int(channel * r)取整数
            nn.ReLU(),
            nn.Linear(int(channel * r), channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 对x进行分支计算权重, 进行全局均值池化
        branch = self.global_avg_pool(x)
        branch = branch.view(branch.size(0), -1)


        # 全连接层得到权重
        weight = self.fc(branch)
        # print(weight.shape)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        b,c = weight.shape
        weight = torch.reshape(weight, (b, c, 1))

        # 乘积获得结果
        scale = weight * x
        return scale

class MTAP(nn.Module):
    def __init__(self,channel):
        super(MTAP, self).__init__()

        self.branch1 = nn.Sequential(
            nn.MaxPool1d(3, 1, padding=1),  # 1.最大池化分支,原文设置的尺寸大小为3, 未说明stride以及padding, 为与原图大小保持一致, 使用(3, 1, 1)
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool1d(3, 1, padding=1),  # 2.平均池化分支, 原文设置的池化尺寸为2, 未说明stride以及padding, 为与原图大小保持一致, 使用(3, 1, 1)
        )

        self.branch3_1 = nn.Sequential(
            nn.Conv1d(channel, int(channel/4), 1),
            nn.Conv1d(int(channel/4), int(channel/4), 3, padding=1),  # 3_1分支, 先用1×1卷积压缩通道维数, 然后使用两个3×3卷积进行特征提取, 由于通道数为3//2, 此时输出维度设为1
            nn.Conv1d(int(channel/4), int(channel/4), 3, padding=1),
        )

        self.branch3_2 = nn.Sequential(
            nn.Conv1d(channel, int(channel*3/4), 1),  # 3_2分支, 由于1×1卷积压缩通道维数减半, 但是这儿维度为3, 上面用的1, 所以这儿输出维度设为2
            nn.Conv1d(int(channel*3/4), int(channel*3/4), 3, padding=1)
        )
        # 注意力机制
        self.branch_SE = SEblock(channel = channel)

        # 初始化可学习权重系数
        # nn.Parameter 初始化的权重, 如果作用到网络中的话, 那么它会被添加到优化器更新的参数中, 优化器更新的时候会纠正Parameter的值, 使得向损失函数最小化的方向优化

        # self.w = nn.Parameter(torch.ones(5))  # 4个分支, 每个分支设置一个自适应学习权重, 初始化为1, nn.Parameter需放入Tensor类型的数据
        # self.w = nn.Parameter(torch.Tensor([0.5, 0.25, 0.15, 0.1]), requires_grad=False)  # 设置固定的权重系数, 不用归一化, 直接乘过去

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

        # 归一化权重
        w1 = torch.exp(weight[0]) / torch.sum(torch.exp(weight))
        w2 = torch.exp(weight[1]) / torch.sum(torch.exp(weight))
        w3 = torch.exp(weight[2]) / torch.sum(torch.exp(weight))
        w4 = torch.exp(weight[3]) / torch.sum(torch.exp(weight))

        # 多特征融合
        x_out = b1 * w1 + b2 * w2 + b3 * w3 + b4 * w4
        # print("特征融合结果:", x_out.shape)
        return x_out


class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.fusion1 = MTAP(32)
        self.fusion2 = MTAP(128)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.g_conv1 = torch.nn.Conv1d(128, 64, 1)
        self.g_conv2 = torch.nn.Conv1d(256, 128, 1)

        self.conv5 = torch.nn.Conv1d(128, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 512, 1)


        self.tarcon = torch.nn.Conv1d(3, 64, 1)
        self.ap1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points

        self.lay1 = torch.nn.BatchNorm1d(64)
        self.lay2 = torch.nn.BatchNorm1d(64)
        self.lay3 = torch.nn.BatchNorm1d(128)
        self.lay4 = torch.nn.BatchNorm1d(128)



    def forward(self, emb, g1, g2, weight1, weight2, weight3, weight4):


        emb = self.fusion1(emb, weight1)
        emb = F.relu(self.lay1(self.e_conv1(emb)))

        g1 = self.fusion2(g1, weight2)
        g = F.relu(self.lay2(self.g_conv1(g1)))  #[1,64,500]


        w1 = (torch.exp(weight3[0]) ) / (torch.sum(torch.exp(weight3)) )
        w2 = torch.exp(weight3[1]) / (torch.sum(torch.exp(weight3)) )
        pointfeat_1 = emb * w1 + g * w2  # 64


        emb = F.relu(self.lay3(self.e_conv2(emb)))
        g = F.relu(self.lay4(self.g_conv2(g2)))


        w3 = (torch.exp(weight4[0])) / (torch.sum(torch.exp(weight4)) )
        w4 = torch.exp(weight4[1]) / (torch.sum(torch.exp(weight4)) )
        pointfeat_2 = emb * w3 + g * w4 #64


        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        # x = self.bn3(x)
        ap_x = self.ap1(x)



        ap_x = ap_x.view(-1, 512, 1).repeat(1, 1, self.num_points)

        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1)  # 128 + 128 + 512



class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        # def __init__(self, num_points, num_obj):
        super(PoseNet, self).__init__()
        self.num_points = num_points

        self.graph_net = DGCNN_cls(1024, 10, 0.5, 500)
        self.cnn = ModifiedResnet()
        self.w = nn.Parameter(torch.ones(4))
        self.w1 = nn.Parameter(torch.ones(4))
        self.w2 = nn.Parameter(torch.ones(2))
        self.w3 = nn.Parameter(torch.ones(2))
        self.feat = PoseNetFeat(num_points)

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




        ap_x = self.feat(emb, x1, x2, self.w, self.w1, self.w2, self.w3)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))


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
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.fusion1 = MTAP(32)
        self.fusion2 = MTAP(128)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.g_conv1 = torch.nn.Conv1d(128, 64, 1)
        self.g_conv2 = torch.nn.Conv1d(256, 128, 1)


        self.conv5 = torch.nn.Conv1d(192, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 512, 1)

        self.tarcon = torch.nn.Conv1d(3, 64, 1)
        self.ap1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points

        self.lay1 = torch.nn.BatchNorm1d(64)
        self.lay2 = torch.nn.BatchNorm1d(64)
        self.lay3 = torch.nn.BatchNorm1d(128)
        self.lay4 = torch.nn.BatchNorm1d(128)


    def forward(self, emb, g1, g2, weight1, weight2, weight3, weight4):
        emb = self.fusion1(emb, weight1)
        emb = F.relu(self.lay1(self.e_conv1(emb)))

        g1 = self.fusion2(g1, weight2)
        g = F.relu(self.lay2(self.g_conv1(g1)))  # [1,64,500]


        w1 = (torch.exp(weight3[0])) / (torch.sum(torch.exp(weight3)))
        w2 = torch.exp(weight3[1]) / (torch.sum(torch.exp(weight3)))
        pointfeat_1 = emb * w1 + g * w2  # 64


        emb = F.relu(self.lay3(self.e_conv2(emb)))
        g = F.relu(self.lay4(self.g_conv2(g2)))


        w3 = (torch.exp(weight4[0])) / (torch.sum(torch.exp(weight4)))
        w4 = torch.exp(weight4[1]) / (torch.sum(torch.exp(weight4)))
        pointfeat_2 = emb * w3 + g * w4

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)


        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        # x = self.bn3(x)
        ap_x = self.ap1(x)


        ap_x = ap_x.view(-1, 512)
        # print("ap_x",ap_x.shape)
        return ap_x


class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat1 = PoseRefineNetFeat(num_points)
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

        self.num_obj = num_obj


    def forward(self, emb, x, R, obj):
        bs = x.size()[0]
        x1 = x.transpose(2, 1).contiguous()
        x1,x2,_ = self.graph_net(x1)
        # x1 = x1.transpose(2, 1).contiguous()

        ap_x = self.feat1(emb, x1, x2, self.w_r, self.w_r1, self.w_r2, self.w_r3)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        # print('rx',rx.shape)
        rx = self.conv1_r_dr(rx)
        tx = self.conv1_t_dr(tx)

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

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
