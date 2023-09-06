#!/home/z_t_h_/Workspace/libs/anaconda3/bin/python3.9
# -*- coding: UTF-8 -*-
import sys
import cv2
import os
import torch
import math
import numpy as np
import logging
import threading
import torch.nn.functional as F
import torchvision
import copy
import random
import threading
import time
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tfs
from torch import nn
from torch import optim
from torch.autograd import Variable
from datetime import datetime
from PIL import Image
from skimage import io as skiio
from skimage import filters as skf
from data_loaders import DDFF12Loader, FoD500Loader
from networks import WDFVNet
from tools import ImgsAndLossHelper
from loss_func import SSIM
from eval import eval_DDFF12, FoD_test
#from ignite.metrics import SSIM
#from ignite.engine import engine

CUDA_ID = 0

DEBUG = False
FORCE_SHOW_IMG = False
# !!! setting both train_with_fod and train_with_ddff to True is undefined behaviour
train_with_ddff = True
train_with_fod = False


def ConfigLogging():
    LOG_FORMAT = "[%(levelname)s][%(asctime)s.%(msecs)03d] %(message)s"
    DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"

    # logging.basicConfig(filename='my.log', level=logging.DEBUG,
    #                    format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logging.basicConfig(level=logging.DEBUG,
                        format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    plt.set_loglevel("info")


def Loss(l1_criterion, l2_criterion, ssim_criterion, ssim_criterions,
         level, predss, stdss, costss, label, focal_dists, epoch):

    # ---------
    max_val = torch.where(focal_dists >= 100, torch.zeros_like(
        focal_dists), focal_dists)  # exclude padding value
    min_val = torch.where(focal_dists <= 0, torch.ones_like(
        focal_dists)*10, focal_dists)  # exclude padding value
    valid_mask = (label >= min_val.min(dim=1)[
        0].view(-1, 1, 1, 1)) & (label <= max_val.max(dim=1)[0].view(-1, 1, 1, 1))
    valid_mask.detach_()
    # ----

    # logging.debug("focal_dists[0]={}, max_val={}, min_val={}".format(
    #     focal_dists[0], max_val, min_val))
    # logging.debug("mask.ones={}".format(torch.count_nonzero(mask)))

    level_weight = [16., 8., 4., 2., 1.]

    loss = 0
    ssim_loss = 0
    l1_loss = 0
    total_weight = 0.
    alpha = 0.025
    [pred_2s, pred_4s, pred_8s, pred_16s, pred_32s] = predss
    for l in range(1):
        preds = [pred_2s[l], pred_4s[l], pred_8s[l],
                 pred_16s[l], pred_32s[l]]  # 不使用深监督不行
        for i, pred in enumerate(preds):
            if pred is None:
                continue

            # logging.debug("label.max={}, label.min={}, pred.max={}, pred.min={}".format(
            #     label.max(), label.min(), pred.max(), pred.min()))

            cur_l1_loss = l1_criterion(pred[valid_mask], label[valid_mask])
            cur_l2_loss = l2_criterion(pred[valid_mask], label[valid_mask])
            cur_ssim_loss = ssim_criterion(label, pred, valid_mask)

            # la_loss = la_criterions(label, pred)

            # 实验结果，前50轮使用SSIM+L1最终MSE=4.41e-4，略好于前20轮使用的4.46e-4
            # 实验结果：前50轮使用MS_SSIM后，效果似乎不如单SSIM，最终4.63e-4@e160，对比4.41e-4@160
            if epoch < 233 * 3:
                cur_ms_ssim_loss = MS_SSIM(
                    ssim_criterions, label, pred, valid_mask)
                cur_loss = cur_ms_ssim_loss
                # cur_loss = cur_l1_loss.mean() + cur_ssim_loss
            else:
                # 48轮后使用[3,7,9]可降到1.6e-4，67轮后仅使用smooth_l1可降到1.3e-4
                cur_loss = cur_l1_loss.mean()

            loss = loss + level_weight[i] * cur_loss
            total_weight = total_weight + level_weight[i]

            if i == 0 and l + 1 == 1:  # level:
                l1_loss = l1_loss + cur_l1_loss.mean()
                ssim_loss = ssim_loss + cur_ssim_loss
    # 权重归一化，用于适应不同level的情况
    loss = loss / total_weight

    cost_loss = 0
    total_weight = 0
    cost_label = torch.zeros_like(costss[0][0]).float()
    b, d, h, w = cost_label.shape

    for d_idx in range(d):
        mask = valid_mask & (label < focal_dists[0, d_idx])
        mask = mask.squeeze(1)
        cost_label[:, d_idx, :, :][mask] = 1
    cost_label[:, :-1, :, :] = cost_label[:, 1:, :, :]
    cost_label[:, -1, :, :] = 0

    # f, axarr = plt.subplots(6)
    # axarr[0].imshow(label[0, 0, :, :].cpu().numpy(), cmap=plt.cm.jet)
    # for idx in range(5):
    #     axarr[idx+1].imshow(
    #         cost_label[0, idx, :, :].cpu().numpy(), cmap=plt.cm.jet)
    # plt.show()

    [cost_2s, cost_4s, cost_8s, cost_16s, cost_32s] = costss
    for l in range(1):
        costs = [cost_2s[l], cost_4s[l], cost_8s[l],
                 cost_16s[l], cost_32s[l]]  # 不使用深监督不行
        for i, cost in enumerate(costs):
            if cost is None:
                continue

            cur_loss = (cost * cost_label) ** 2
            cost_loss = cost_loss + level_weight[i] * cur_loss.mean()
            total_weight = total_weight + level_weight[i]
    cost_loss = cost_loss / total_weight

    # loss = loss + cost_loss
    return loss, l1_loss, ssim_loss, cost_loss


def CreateOptimizer(depth_net, lr):
    params = list(depth_net.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=64, gamma=0.1)
    return optimizer, scheduler


def MS_SSIM(ssims, label, pred, valid_mask, max_ssim_cnt=100):
    res = 0
    for idx, ssim in enumerate(ssims):
        if idx > max_ssim_cnt:
            break
        cur_res = ssim(pred, label, valid_mask)
        res = res + cur_res
        # logging.debug("MS_SSIM[{}]={}".format(idx, cur_res))
    return res


def Train():
    global CUDA_ID
    global train_with_ddff
    global train_with_fod

    assert train_with_ddff != train_with_fod, "Cannot set both train_with_ddff and train_with_fod to True or to False for now"

    rand_seed = 2333
    random.seed(rand_seed)
    os.environ['PYTHONHASHSEED'] = str(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True

    logging.debug(">>>>>> train begin <<<<<<<<")
    epoch_num = 233 * 3
    batch_size = 8
    lr = 1e-4
    load_epoch = -1
    use_scheduler = False
    net_param_all_created = False
    level = 5  # unused
    use_diff = True  # unused
    logging.debug("Train: params={}".format(locals()))

    #################################### datasets ####################################
    ddff_database = "../data_sets/ddff-dfv/ddff-dataset-trainval.h5"
    ddff_train_dataset = DDFF12Loader(
        ddff_database, stack_key="stack_train", disp_key="disp_train", n_stack=5, min_disp=0.02, max_disp=0.28)
    ddff_eval_dataset = DDFF12Loader(
        ddff_database, stack_key="stack_val", disp_key="disp_val", n_stack=5, min_disp=0.02, max_disp=0.28, b_test=False)

    fod_database = "../data_sets/FoD500/"
    fod_train_dataset, fod_eval_dataset = FoD500Loader(
        fod_database, n_stack=5, scale=0.2)

    train_dataset = []
    if train_with_ddff:
        logging.debug("train with ddff dataset")
        train_dataset += [ddff_train_dataset]
        data_path = ddff_database
    elif train_with_fod:
        logging.debug("train with fod dataset")
        train_dataset += [fod_train_dataset]
        data_path = fod_database

    assert len(train_dataset) > 0

    train_dataset = torch.utils.data.ConcatDataset(train_dataset)

    num_workers = 12 if train_with_ddff else 0
    train_data_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, drop_last=True, shuffle=False if DEBUG else True, num_workers=num_workers)
    eval_data_loader = DataLoader(
        dataset=fod_eval_dataset if train_with_fod else ddff_eval_dataset,
        batch_size=1, shuffle=False, num_workers=0)

    logging.debug(">>>>>> data loaded <<<<<<<<")

    ################################# load network #####################################
    if load_epoch >= 0:
        net_param_all_created = True
        saved_params = torch.load(
            "../saved_models/WDFV_e{:0>3d}.depth_net.mdl".format(load_epoch), map_location={'cuda:3': 'cuda:0'})
        depth_net = saved_params['net']
        optimizer = saved_params['opt']
        scheduler = saved_params['sch']
        lr = saved_params['lr']
        logging.info(">>>>>>> network e{} loaded <<<<<<<<<".format(load_epoch))
    else:
        depth_net = WDFVNet.WDFVNet(level, use_diff)
        optimizer, scheduler = CreateOptimizer(depth_net, lr)
    if CUDA_ID >= 0:
        depth_net = depth_net.cuda(CUDA_ID)
    else:
        depth_net = depth_net.cpu()
    ##################################################################################
    logging.debug(
        ">>>>>> net prepared, level={} <<<<<<<<".format(depth_net.level))

    ############################# loss functions #######################################
    l1_criterion = torch.nn.SmoothL1Loss(reduction="none")
    l2_criterion = torch.nn.MSELoss()
    # setup ssim criterion
    ssim_criterion = SSIM.SSIM(CUDA_ID, kernel_size=11)
    ssim_criterions = []
    ssim_kernels = [1, 3, 5, 9, 17]  # , 33]
    for idx, ssim_kernel in enumerate(ssim_kernels):
        ssim_criterions.append(
            SSIM.SSIM(CUDA_ID, kernel_size=ssim_kernel))
    ##################################################################################

    # setup finish

    start_time = datetime.now()
    pre_time = start_time
    if DEBUG:
        imgs_and_loss_helper = ImgsAndLossHelper.ImgsAndLossHelper(
            "../saved_imgs/", data_type="train", stride=1, show_img=True)
    else:
        imgs_and_loss_helper = ImgsAndLossHelper.ImgsAndLossHelper(
            "../saved_imgs/", data_type="train", stride=10, show_img=True if FORCE_SHOW_IMG else False)

    eval_thread = None
    for epoch in trange(load_epoch + 1, epoch_num):
        logging.debug(">>>>>>> begin epoch {} <<<<<<<<".format(epoch))
        epoch_loss = 0
        l1_loss = 0
        ssim_loss = 0
        cost_loss = 0
        data_len = len(train_data_loader)
        depth_net.train()

        logging.debug("")

        try:
            for batch_idx, (batch_imgs, batch_label, focal_dists) in enumerate(train_data_loader):
                # logging.debug("batch_imgs.shape={}".format(batch_imgs.shape))
                optimizer.zero_grad()

                # if batch_idx > 0:
                #     break

                batch_imgs = Variable(batch_imgs.cuda(CUDA_ID))
                batch_label = Variable(batch_label.cuda(CUDA_ID))
                focal_dists = Variable(focal_dists.cuda(CUDA_ID))

                diff_imgs, predss, stdss, costss = depth_net(
                    batch_imgs, focal_dists)
                loss, cur_l1_loss, cur_ssim_loss, cur_cost_loss = Loss(
                    l1_criterion, l2_criterion, ssim_criterion, ssim_criterions,
                    level, predss, stdss, costss, batch_label, focal_dists, epoch)

                epoch_loss += loss.detach().cpu().numpy()
                l1_loss += cur_l1_loss.detach().cpu().numpy()
                ssim_loss += cur_ssim_loss.detach().cpu().numpy()
                cost_loss += cur_cost_loss.detach().cpu().numpy()

                loss.backward()

                # 先backward()再clip_grad? clip多少怎么确定?似乎是能够防止变全0的操作?
                # nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()

                # for name, parms in depth_net.named_parameters():
                #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
                #           ' -->grad_value max:', parms.grad.max() if parms.grad is not None else None,
                #           ' -->grad_value min:', parms.grad.min() if parms.grad is not None else None)

                ########################## show imgs ##############################
                batch_imgs = diff_imgs
                # l = net.level
                [pred_2s, pred_4s, pred_8s, pred_16s, pred_32s] = predss
                preds = [pred_2s[0], pred_4s[0],
                         pred_8s[0], pred_16s[0], pred_32s[0]]
                [std_2s, std_4s, std_8s, std_16s, std_32s] = stdss
                stds = [std_2s[0], std_4s[0],
                        std_8s[0], std_16s[0], std_32s[0]]
                single_preds = torch.stack(
                    (preds[0][0], preds[1][0], preds[2][0], (preds[0][0] - preds[1][0]).abs()+0.02, stds[0][0], stds[1][0]), dim=0)
                imgs_and_loss_helper.Update(
                    batch_imgs[0], batch_label[0], single_preds, focal_dists[0], loss.data, epoch, batch_idx, data_len)
                ###################################################################

                batch_idx += 1
                # cv2.waitKey(0)

                if not net_param_all_created:
                    net_param_all_created = True
                    logging.info(
                        "net param NOW all created, no need reinit optimizer && scheduler")
                    # optimizer, scheduler = CreateOptimizer(depth_net, lr)
        except:
            # num_workers > 0 偶尔会在win下会抛出raise empty异常
            logging.warning("Unexpected error:", sys.exc_info()[0])
            epoch -= 1
            raise
            continue

        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        epoch_loss /= batch_idx
        l1_loss /= data_len
        ssim_loss /= data_len
        cost_loss /= data_len
        logging.info(
            "train loss@epoch{:0>3d}: {:.8f}".format(epoch, epoch_loss))
        logging.info("l1 loss@epoch{:0>3d}: {:.8f}".format(epoch, l1_loss))
        logging.info("ssim loss@epoch{:0>3d}: {:.8f}".format(epoch, ssim_loss))
        logging.info("cost loss@epoch{:0>3d}: {:.8f}".format(epoch, cost_loss))
        logging.info("lr@epoch{:0>3d}: {:.8f}".format(epoch, cur_lr))

        mdl_path = "../saved_models/WDFV_e{:0>3d}.depth_net.mdl".format(epoch)
        torch.save({'net': depth_net, 'opt': optimizer,
                    'sch': scheduler, 'lr': cur_lr}, mdl_path)
        fo = open(
            "../saved_models/WDFV_e{:0>3d}.loss".format(epoch), "w")
        fo.write("train loss@epoch{:0>3d}: {:.8f}\n".format(epoch, epoch_loss))
        fo.write("l1 loss@epoch{:0>3d}: {:.8f}\n".format(epoch, l1_loss))
        fo.write("ssim loss@epoch{:0>3d}: {:.8f}\n".format(epoch, ssim_loss))
        fo.write("cost loss@epoch{:0>3d}: {:.8f}\n".format(epoch, cost_loss))
        fo.write("lr@epoch{:0>3d}: {:.8f}\n".format(epoch, cur_lr))
        fo.close()

        if use_scheduler:
            scheduler.step()

        PROJ_DIR = "../"
        show_img = True if DEBUG or FORCE_SHOW_IMG else False
        stride = 1 if DEBUG else 20
        eval_func = eval_DDFF12 if train_with_ddff else FoD_test
        image_size = (383, 552)
        logging.debug("eval_func={}".format(eval_func.__name__))
        if eval_thread is not None:
            logging.info("begin waiting eval_thread")
            eval_thread.join()
            logging.info("eval_thread finish")
        eval_thread = threading.Thread(target=eval_func.Eval, args=(
            data_path, mdl_path, image_size, PROJ_DIR, epoch, stride, show_img))
        eval_thread.start()
        eval_thread.join()


if __name__ == "__main__":
    ConfigLogging()

    logging.info('>'*10 + "Training WDFV network..." + '<'*10)

    Train()

    logging.info('>'*10 + "FIN" + '<'*10)
