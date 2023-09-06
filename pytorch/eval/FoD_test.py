import sys
sys.path.append("..")  # nopep8

from data_loaders import DDFF12Loader
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
import os
import skimage.filters as skf
import time
import torch
import logging
from torch.autograd import Variable
from tools import ImgsAndLossHelper
from torch.utils.data import DataLoader
from eval import eval_FoD500


# import  matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

'''
Main code for Ours-FV and Ours-DFV test on FoD500 dataset  
'''

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# parser = argparse.ArgumentParser(description='DFVDFF')
# parser.add_argument('--data_path', default='/data/DFF/baseline/defocus-net/data/fs_6/',help='test data path')
# parser.add_argument('--loadmodel', default=None, help='model path')
# parser.add_argument('--outdir', default='./FoD500/',help='output dir')
#
# parser.add_argument('--stack_num', type=int ,default=5, help='num of image in a stack, please take a number in [2, 10], change it according to the loaded checkpoint!')
# parser.add_argument('--use_diff', default=1, choices=[0,1], help='if use differential images as input, change it according to the loaded checkpoint!')
#
# parser.add_argument('--level', type=int, default=4, help='num of layers in network, please take a number in [1, 4]')
# args = parser.parse_args()

# !!! Only for users who download our pre-trained checkpoint, comment the next four line if you are not !!!
# if os.path.basename(args.loadmodel) == 'DFF-DFV.tar' :
#     args.use_diff = 1
# else:
#     args.use_diff = 0


# dataloader
from data_loaders import FoD500Loader

# construct model
# model = DFFNet( clean=False,level=args.level, use_diff=args.use_diff)
# model = nn.DataParallel(model)
# model.cuda()
# ckpt_name = os.path.basename(os.path.dirname(args.loadmodel))# we use the dirname to indicate training setting
#
# if args.loadmodel is not None:
#     pretrained_dict = torch.load(args.loadmodel)
#     pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
#     model.load_state_dict(pretrained_dict['state_dict'],strict=False)
# else:
#     print('run with random init')
# print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def disp2depth(disp):
    dpth = 1 / disp
    dpth[disp == 0] = 0
    return dpth


# def main(image_size = (256, 256)):

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


def Eval(data_path, mdl_path, image_size=(256, 256), PROJ_DIR=None, epoch=None, stride=1, show_img=False):
    imgs_and_loss_helper = ImgsAndLossHelper.ImgsAndLossHelper(
        PROJ_DIR + "saved_imgs/eval/", data_type="eval", stride=stride, show_img=show_img)

    model = torch.load(mdl_path)['net'].cuda()
    model.eval()

    dataset_train, dataset_validation = FoD500Loader(data_path, n_stack=5)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset_validation, num_workers=0, batch_size=1, shuffle=False)

    # metric prepare
    test_num = len(dataloader)
    time_list = []
    std_sum = 0
    for inx, (img_stack, gt_disp, foc_dist) in enumerate(dataloader):
        # if inx not in  [5, 64,67]:continue
        if inx % 10 == 0:
            logging.debug('processing: {}/{}'.format(inx, test_num))

        img_stack = Variable(torch.FloatTensor(img_stack)).cuda()
        foc_dist = Variable(torch.FloatTensor(foc_dist)).cuda()

        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            # diff_imgs, predss, stdss = model(img_stack, foc_dist)
            diff_imgs, predss, stdss, costss = model(img_stack, foc_dist)
            torch.cuda.synchronize()
            ttime = (time.time() - start_time)
            # print('time = %.2f' % (ttime*1000) )

        [pred_2s, pred_4s, pred_8s, pred_16s, pred_32s] = predss
        preds = [pred_2s[0], pred_4s[0], pred_8s[0], pred_16s[0], pred_32s[0]]
        [std_2s, std_4s, std_8s, std_16s, std_32s] = stdss
        stds = [std_2s[0], std_4s[0], std_8s[0], std_16s[0], std_32s[0]]

        std = stds[0][0].squeeze().cpu().numpy()[
            :image_size[0], :image_size[1]]
        pred_dpth = preds[0][0].squeeze().cpu().numpy()[
            :image_size[0], :image_size[1]]
        gt_dpth = gt_disp.squeeze().numpy()

        std_sum += std.mean()
        cur_loss = ((pred_dpth - gt_dpth) ** 2).mean()

        img_save_pth = os.path.join(
            PROJ_DIR + "saved_imgs/fod_test_res/")  # 'figure_paper'#
        if not os.path.exists(img_save_pth):
            os.makedirs(img_save_pth)

        # save for eval
        img_id = inx + 400
        cv2.imwrite('{}/{}_pred.png'.format(img_save_pth, img_id),
                    (pred_dpth * 10000).astype(np.uint16))
        cv2.imwrite('{}/{}_gt.png'.format(img_save_pth, img_id),
                    (gt_dpth * 10000).astype(np.uint16))

        ########################## show imgs ##############################
        # img_stack = diff_imgs
        single_preds = torch.stack(
            (preds[0][0], preds[1][0], preds[2][0], preds[0][0] - preds[1][0], stds[0][0], stds[1][0]), dim=0)
        imgs_and_loss_helper.Update(
            img_stack[0], gt_disp[0], single_preds, foc_dist[0], cur_loss, epoch, inx, test_num)
        # logging.debug("id{} mse={}".format(inx, metrics[0, 0]))
        # cv2.waitKey(0)
        # SplitAndPred(PROJ_DIR, model, img_stack, foc_dist, disp, epoch, inx, test_num)
        ###################################################################

        # =========== only need for debug ================
        # err map
        # mask = (gt_dpth > 0)  # .float()
        # err = (np.abs(pred_dpth.clip(0,1.5) - gt_dpth.clip(0, 1.5)) * mask).clip(0, 0.3)
        #
        # cv2.imwrite('{}/viz/{}_err.png'.format(img_save_pth, img_id), err * (255/0.3))

        # pred viz
        # MAX_DISP, MIN_DISP = 1.5, 0
        # # pred_disp = pred_disp.squeeze().detach().cpu().numpy()
        # plt.figure()
        # plt.imshow(pred_disp, vmax=MAX_DISP, vmin=MIN_DISP)  # val2uint8(, MAX_DISP, MIN_DISP)
        # plt.axis('off')
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.savefig('{}/viz/{}_pred_viz.png'.format(img_save_pth, img_id), bbox_inches='tight', pad_inches=0)
        # plt.close()
        #
        # # std viz
        # plt.imshow(std.squeeze().detach().cpu().numpy(), vmax=0.5, vmin=0)  # val2uint8(, MAX_DISP, MIN_DISP)
        # plt.axis('off')
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.savefig('{}/viz/{}_std_viz.png'.format(img_save_pth, img_id,  args.level), bbox_inches='tight', pad_inches=0)
        #
        # for i in range(args.stack_num):
        #     MAX_DISP, MIN_DISP = 1, 0
        #     plt.imshow(focusMap[i].squeeze().detach().cpu().numpy(), vmax=MAX_DISP,
        #                vmin=MIN_DISP, cmap='jet')  # val2uint8(, MAX_DISP, MIN_DISP)
        #     plt.axis('off')
        #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #     plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        #     plt.margins(0, 0)
        #     plt.savefig('{}/{}_{}_prob_dist.png'.format(img_save_pth, img_id, i), bbox_inches='tight', pad_inches=0)

        # time
        time_list.append('{} {}\n'.format(img_id, ttime))

    logging.debug("avgUnc. = {}".format(std_sum / len(dataloader)))
    with open('{}/runtime.txt'.format(img_save_pth), 'w') as f:
        for line in time_list:
            f.write(line)

    eval_FoD500.EvalMtx(img_save_pth)


if __name__ == '__main__':
    ConfigLogging()

    logging.debug("Start eval")
    epoch = 233*3
    stride = 20  # save image once every 20 step
    show_img = False
    data_path = "../../data_sets/FoD500/"
    # mdl_path = "../../saved_models/WDFV_e{:0>3d}.depth_net.mdl".format(epoch)
    mdl_path = "../../saved_models/WDFV-FoD_129.depth_net.mdl"

    Eval(data_path, mdl_path, PROJ_DIR="../../",
         epoch=epoch, stride=stride, show_img=show_img)
