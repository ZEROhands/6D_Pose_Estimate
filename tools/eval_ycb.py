import argparse
import copy
import numpy as np
from PIL import Image
import scipy.io as scio
import numpy.ma as ma
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_matrix, quaternion_from_matrix
import os
import spherical
import cv2




parser = argparse.ArgumentParser()
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser.add_argument('--dataset_root', type=str, default='',
                    help='dataset root dir')
parser.add_argument('--model', type=str, default='',
                    help='resume PoseNet model')
parser.add_argument('--refine_model', type=str,
                    default='',
                    help='resume PoseRefineNet model')
opt = parser.parse_args()


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


# symmetry_obj_idx = [12, 15, 18, 19, 20]
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
cam_cx = 312.9869
cam_cy = 241.3109
cam_fx = 1066.778
cam_fy = 1067.487
cam_scale = 10000.0
num_obj = 21
img_width = 480
img_length = 640
num_points = 1000
num_points_mesh = 500
iteration = 8
bs = 1
dataset_config_dir = 'datasets/ycb/dataset_config'
ycb_toolbox_dir = 'YCB_toolbox'
result_wo_refine_dir = ''
result_refine_dir = ''
def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax




estimator = PoseNet(num_points=num_points, num_obj=num_obj)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()

refiner = PoseRefineNet(num_points=num_points, num_obj=num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(opt.refine_model))
refiner.eval()

testlist = []
input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    testlist.append(input_line)
input_file.close()
print(len(testlist))

class_file = open('{0}/classes.txt'.format(dataset_config_dir))
class_id = 1
cld = {}
while 1:
    class_input = class_file.readline()
    if not class_input:
        break
    class_input = class_input[:-1]

    input_file = open('{0}/models/{1}/points.xyz'.format(opt.dataset_root, class_input))
    cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1]
        input_line = input_line.split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    input_file.close()
    cld[class_id] = np.array(cld[class_id])
    class_id += 1




for now in range(0, 2949):
    img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
    depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, testlist[now])))

    label = np.array(Image.open('{0}/{1}-label.png'.format(opt.dataset_root, testlist[now])))
    meta = scio.loadmat('{0}/{1}-meta.mat'.format(opt.dataset_root, testlist[now]))
    obj = meta['cls_indexes'].flatten().astype(np.int32)
    lst = obj

    my_result_wo_refine = []
    my_result = []

    for idx in range(len(lst)):
        itemid = lst[idx]
        try:

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
            mask = mask_label * mask_depth

            rmin, rmax, cmin, cmax = get_bbox(mask_label)

            a = mask[rmin:rmax, cmin:cmax]


            tmp = np.zeros(shape=(a.shape[0], a.shape[1], 3))
            for i in range(3):
                tmp[:, :, i] = a
            a = cv2.resize(tmp, (80, 80))
            a = a[:, :, 0]

            choose = a.flatten().nonzero()[0]
            b = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            if len(choose) == 0:
                continue
            if len(choose) > num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')
                # change2
            depth_masked = depth[rmin:rmax, cmin:cmax]
            tmp = np.zeros(shape=(depth_masked.shape[0], depth_masked.shape[1], 3))
            for i in range(3):
                tmp[:, :, i] = depth_masked
                # print('depth_masked.shape:{}'.format(tmp.shape))
            depth_masked = cv2.resize(tmp, (80, 80))
                # print('depth_masked.shape:{}'.format(depth_masked.shape))
            depth_masked = depth_masked[:, :, 0]
            depth_masked = depth_masked.flatten()[choose][:, np.newaxis].astype(np.float32)
                # depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            b = xmap[rmin:rmax, cmin:cmax]
            tmp = np.zeros(shape=(b.shape[0], b.shape[1], 3))
            for i in range(3):
                tmp[:, :, i] = b
            b = cv2.resize(tmp, (80, 80))
            b = b[:, :, 0]

            c = ymap[rmin:rmax, cmin:cmax]
            tmp = np.zeros(shape=(c.shape[0], c.shape[1], 3))
            for i in range(3):
                tmp[:, :, i] = c
            c = cv2.resize(tmp, (80, 80))
            c = c[:, :, 0]
                # depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = b.flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = c.flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)

            img_masked = np.array(img)[:, :, :3]
            img_masked = np.transpose(img_masked, (2, 0, 1))
            img_masked = img_masked[:, rmin:rmax, cmin:cmax]

                # change 3
            img_masked = img_masked.transpose(1, 2, 0)
            img_masked = cv2.resize(img_masked, (80, 80))
            img_masked = img_masked.transpose(2, 0, 1)

            cloud = torch.from_numpy(cloud.astype(np.float32))
            choose = torch.LongTensor(choose.astype(np.int32))
            img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
            index = torch.LongTensor([itemid - 1])

                # cloud = Variable(cloud).cuda()
            choose = Variable(choose).cuda()
                # img_masked = Variable(img_masked).cuda()
            index = Variable(index).cuda()
                # print(cloud.shape)[1000,3]
                # cloud = cloud.view(1, num_points, 3)

            img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

            cloud, R_train = spherical.get_Spherical_coordinate(cloud.cpu(), normalized=True)
            cloud = torch.from_numpy(cloud.astype(np.float32)).view(1, num_points, 3).cuda()
                # cloud = cloud = Variable(cloud).cuda()
            R_train = torch.from_numpy(R_train.astype(np.float32)).cuda()
                # print(cloud.shape)[1,3,3]
            pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, R_train, choose, index)

            cloud = zhijiao(cloud, R_train).cuda()
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

            pred_c = pred_c.view(bs, num_points)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(bs * num_points, 1, 3)
            points = cloud.view(bs * num_points, 1, 3)

            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
            my_pred = np.append(my_r, my_t)
            my_result_wo_refine.append(my_pred.tolist())

            for ite in range(0, iteration):
                T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points,
                                                                                                     1).contiguous().view(1,
                                                                                                                          num_points,
                                                                                                                          3)
                my_mat = quaternion_matrix(my_r)
                R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                my_mat[0:3, 3] = my_t

                new_cloud = torch.bmm((cloud - T), R).contiguous()
                pred_r, pred_t = refiner(emb, new_cloud, R_train, index)
                pred_r = pred_r.view(1, 1, -1)
                pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                my_r_2 = pred_r.view(-1).cpu().data.numpy()
                my_t_2 = pred_t.view(-1).cpu().data.numpy()
                my_mat_2 = quaternion_matrix(my_r_2)

                my_mat_2[0:3, 3] = my_t_2

                my_mat_final = np.dot(my_mat, my_mat_2)
                my_r_final = copy.deepcopy(my_mat_final)
                my_r_final[0:3, 3] = 0
                my_r_final = quaternion_from_matrix(my_r_final, True)
                my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                my_pred = np.append(my_r_final, my_t_final)
                my_r = my_r_final
                my_t = my_t_final

            my_result.append(my_pred.tolist())
        except ZeroDivisionError:
            print("PoseCNN Detector Lost {0} at No.{1} keyframe".format(itemid, now))
            my_result_wo_refine.append([0.0 for i in range(7)])
            my_result.append([0.0 for i in range(7)])
    print(lst)
    scio.savemat('{0}/{1}.mat'.format(result_wo_refine_dir, '%04d' % now),
                     {'poses': my_result_wo_refine,'rois':lst})
    scio.savemat('{0}/{1}.mat'.format(result_refine_dir, '%04d' % now),
                     {'poses': my_result,'rois':lst})
    print("Finish No.{0} keyframe".format(now))