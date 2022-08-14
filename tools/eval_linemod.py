import argparse
import time
import sys
sys.path.append("../")
import numpy as np
import yaml
import copy
import torch.utils.data
from torch.autograd import Variable
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
# from lib.network8 import PoseNet
from lib.network import  PoseNet,PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')

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
num_objects = 13
objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
num_points = 500

iteration =8
bs = 1
dataset_config_dir = '../datasets/linemod/dataset_config'
output_result_dir = '../experiments/eval_result/linemod'
knn = KNearestNeighbor(1)

estimator = PoseNet(num_points = num_points, num_obj = num_objects)
estimator.cuda()
refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
refiner.cuda()
estimator.load_state_dict(torch.load(opt.model),strict=False)
refiner.load_state_dict(torch.load(opt.refine_model),strict=False)
estimator.eval()
refiner.eval()

testdataset = PoseDataset_linemod('eval', num_points, False, opt.dataset_root, 0.0, True, is_spherical=True)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)

sym_list = testdataset.get_sym_list()
num_points_mesh = testdataset.get_num_points_mesh()
criterion = Loss(num_points_mesh, sym_list)
criterion_refine = Loss_refine(num_points_mesh, sym_list)

diameter = []
meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
meta = yaml.load(meta_file,Loader=yaml.FullLoader)
for obj in objlist:
    diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)

num_first = []
success_count = [0 for i in range(num_objects)]

num_count = [0 for i in range(num_objects)]
localtime = time.asctime( time.localtime(time.time()))
fw = open('{0}/{1}_{2}_{3}_eval_result_logs.txt'.format(output_result_dir,localtime.split(' ')[1],localtime.split(' ')[3],localtime.split(' ')[4]), 'w')

for i, data in enumerate(testdataloader, 0):
    points, R, choose, img, target, model_points, idx= data

    if len(points.size()) == 2:
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
        continue
    points,R, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                           Variable(R).cuda(), \
                                                           Variable(choose).cuda(), \
                                                         Variable(img).cuda(), \
                                                         Variable(target).cuda(), \
                                                         Variable(model_points).cuda(), \
                                                         Variable(idx).cuda()
        # st_time = time.time()

    pred_r, pred_t, pred_c, emb = estimator(img, points, R, choose, idx)
        # print('forward_posnet:',(time.time()-st_time)*1000)
    points = zhijiao(points, R).cuda()
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
    pred_c = pred_c.view(bs, num_points)
    how_max, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(bs * num_points, 1, 3)

    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
    my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
    my_pred = np.append(my_r, my_t)


    model_points = model_points[0].cpu().detach().numpy()
    target = target[0].cpu().detach().numpy()
    for ite in range(0, iteration):
        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
        my_mat = quaternion_matrix(my_r)
        R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
        my_mat[0:3, 3] = my_t

        new_points = torch.bmm((points - T), R).contiguous()
        st1_time = time.time()
        pred_r, pred_t = refiner(emb, new_points, R, idx)
            # print('forward_refine_posnet_onetime:', (time.time() - st1_time) * 1000)
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


        my_r1 = quaternion_matrix(my_r)[:3, :3]
        pred = np.dot(model_points, my_r1.T) + my_t

        if idx[0].item() in sym_list:
            pred1 = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
            target1 = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
            inds = KNearestNeighbor.apply(target1.unsqueeze(0), pred1.unsqueeze(0))
            target1 = torch.index_select(target1, 1, inds.view(-1) - 1)
            dis = torch.mean(torch.norm((pred1.transpose(1, 0) - target1.transpose(1, 0)), dim=1), dim=0).item()
        else:
            dis = np.mean(np.linalg.norm(pred - target, axis=1))
        if dis < diameter[0]:
            break

    if dis < diameter[idx[0].item()]:
        success_count[idx[0].item()] += 1
        print('No.{0} Pass! Distance: {1} '.format(i, dis))
        fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis))
    else:
        print('No.{0} NOT Pass! Distance: {1} '.format(i, dis))
        fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis))
    num_count[idx[0].item()] += 1

for i in range(num_objects):
    print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i]) / num_count[i]))
print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))

fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
fw.write("posenet_model :{0}\nrefinenet_model :{1}".format(opt.model,opt.refine_model))
fw.close()