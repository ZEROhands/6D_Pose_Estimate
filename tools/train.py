# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------
import sys

sys.path.append("../")

import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger



# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='linemod', help='ycb or linemod')
parser.add_argument('--dataset_root', type=str, default='',
                    help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--lr', default=0.00001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.011, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03,
                    help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default=2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--resume_refinenet', type=str, default='', help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--is_spherical', type=int, default=True, help='trans to spherical_coordiantes')
parser.add_argument('--start_refine_dire', type=int, default=False, help='directly use resume_posenet into refine')
opt = parser.parse_args()

print("预设值dataset是{0}\nresume是{1}".format(opt, opt.resume_posenet))




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


def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'ycb':
        opt.num_objects = 21  # number of object classes in the dataset
        opt.num_points = 1000  # number of points on the input pointcloud
        opt.outf = 'trained_models/ycb'  # folder to save trained models
        opt.log_dir = '../experiments/logs/ycb'  # folder to save logs
        opt.repeat_epoch = 1  # number of repeat times for one epoch training
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = 'trained_models/linemod'
        opt.log_dir = '../experiments/logs/linemod'
        opt.repeat_epoch = 20
    else:
        print('Unknown dataset')
        return

    estimator = PoseNet(opt.num_points, opt.num_objects)
    estimator.cuda()
    refiner = PoseRefineNet(opt.num_points, opt.num_objects)
    refiner.cuda()

    print(opt.dataset)
    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)),strict=False)

    if opt.resume_refinenet != '':
        refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)),strict=False)
        opt.refine_start = True
        opt.decay_start = True
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        opt.batch_size = int(opt.batch_size / opt.iteration)
        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    elif opt.start_refine_dire:
        opt.refine_start = True
        opt.decay_start = True
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        opt.batch_size = int(opt.batch_size / opt.iteration)
        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)



    if opt.dataset == 'ycb':
        dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start,
                                  opt.is_spherical)
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans,
                                      opt.refine_start, opt.is_spherical)
    dataloader = torch.utils.data.DataLoader(dataset,num_workers=opt.workers, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start,
                                       opt.is_spherical)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start,
                                           opt.is_spherical)
    testdataloader = torch.utils.data.DataLoader(test_dataset,num_workers=opt.workers,  batch_size=opt.test_batch_size, shuffle=False,
                                                 drop_last=True)

    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()
    # opt.sym_list = test_dataset.get_sym_list()
    # opt.num_points_mesh = test_dataset.get_num_points_mesh()

    # print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh, opt.sym_list)
    criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

    best_test = np.Inf

    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()


    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0
        if opt.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()
        optimizer.zero_grad()

        for rep in range(opt.repeat_epoch):
            # train_count = 0
            for i, data in enumerate(dataloader, 0):
                points, R, choose, img, target, model_points, idx = data
                points, R, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                                    Variable(R).cuda(), \
                                                                    Variable(choose).cuda(), \
                                                                    Variable(img).cuda(), \
                                                                    Variable(target).cuda(), \
                                                                    Variable(model_points).cuda(), \
                                                                    Variable(idx).cuda(), \
                    # points_ori = points
                pred_r, pred_t, pred_c, emb = estimator(img, points, R, choose, idx)

                points = zhijiao(points, R)
                # print("pred_r",pred_r.shape)
                loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points,
                                                              opt.w, opt.refine_start)

                if opt.refine_start:
                    for ite in range(0, opt.iteration):
                        pred_r, pred_t = refiner(emb, new_points, R, idx)

                        dis, new_points, new_target = criterion_refine(pred_r,
                                                                        pred_t,
                                                                        new_target,
                                                                        model_points,
                                                                        idx,
                                                                        new_points)
                        dis.mean().backward()
                    train_dis_avg += dis.sum().item()
                       #ew_points = new_points1.contiguous()

                else:
                    loss.backward()
                    train_dis_avg += dis.sum().item()


                # print(dis.shape)
                train_count += opt.batch_size

                if train_count % opt.batch_size == 0:
                    logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4} LOSS:{5}'.format(
                        time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch,
                        int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size, loss))
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0

                if train_count != 0 and train_count % 1000 == 0:
                    if opt.refine_start:
                        torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(opt.outf))
                    else:
                        torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()
        refiner.eval()

        for j, data in enumerate(testdataloader, 0):
            points, R, choose, img, target, model_points, idx = data

            # print("-----\n{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n------".format(points.shape,choose.shape,img.shape,target.shape,model_points.shape,idx.shape))
            points, R, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                                Variable(R).cuda(), \
                                                                Variable(choose).cuda(), \
                                                                Variable(img).cuda(), \
                                                                Variable(target).cuda(), \
                                                                Variable(model_points).cuda(), \
                                                                Variable(idx).cuda(), \
                # points_ori = points
            pred_r, pred_t, pred_c, emb = estimator(img, points, R, choose, idx)
            points = zhijiao(points, R)
            # points = paixu(points, num)
            _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w,
                                                       opt.refine_start)

            if opt.refine_start:
                for ite in range(0, 2):
                    pred_r, pred_t = refiner(emb, new_points, R, idx)



                    dis, new_points, new_target = criterion_refine(pred_r,
                                                                                   pred_t,
                                                                                   new_target,
                                                                                   model_points,
                                                                                   idx,
                                                                                   new_points)

                test_dis += dis.sum().item()

                    # dis.backward()


            else:
                test_dis += dis.sum().item()



            logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis))

            test_count += opt.test_batch_size

        test_dis = test_dis / test_count
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))
        if test_dis <= best_test:
            best_test = test_dis
            if opt.refine_start:
                torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            else:
                torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

        if best_test < opt.refine_margin and not opt.refine_start:
            opt.refine_start = True
            opt.batch_size = int(opt.batch_size / opt.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)

            if opt.dataset == 'ycb':
                dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans,
                                          opt.refine_start, opt.is_spherical)
            elif opt.dataset == 'linemod':
                dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans,
                                              opt.refine_start, opt.is_spherical)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
            if opt.dataset == 'ycb':
                test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start,
                                               opt.is_spherical)
            elif opt.dataset == 'linemod':
                test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0,
                                                   opt.refine_start, opt.is_spherical)
            testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False)

            opt.sym_list = dataset.get_sym_list()
            opt.num_points_mesh = dataset.get_num_points_mesh()

            print(
                '>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(
                    len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

            criterion = Loss(opt.num_points_mesh, opt.sym_list)
            criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)


if __name__ == '__main__':
    main()
