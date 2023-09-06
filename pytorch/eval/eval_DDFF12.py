import sys
sys.path.append("..")  # nopep8

from tools import ImgsAndLossHelper
from torch.autograd import Variable
import logging
import torch
import time
import skimage.filters as skf
import os
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from data_loaders import DDFF12Loader


# import matplotlib
# matplotlib.use('TkAgg')


'''
Code for Ours-FV and Ours-DFV evaluation on DDFF-12 dataset  
'''

# # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# parser = argparse.ArgumentParser(description='DFVDFF')
# parser.add_argument(
#     '--data_path', default='/data/DFF/my_ddff_trainVal.h5', help='test data path')
# parser.add_argument('--loadmodel', default=None, help='model path')
# parser.add_argument('--outdir', default='./DDFF12/', help='output dir')
#
# parser.add_argument('--max_disp', type=float,
#                     default=0.28, help='maxium disparity')
# parser.add_argument('--min_disp', type=float,
#                     default=0.02, help='minium disparity')
#
# parser.add_argument('--stack_num', type=int, default=5,
#                     help='num of image in a stack, please take a number in [2, 10], change it according to the loaded checkpoint!')
# parser.add_argument('--use_diff', default=1, choices=[
#                     0, 1], help='if use differential images as input, change it according to the loaded checkpoint!')
#
# parser.add_argument('--level', type=int, default=4,
#                     help='num of layers in network, please take a number in [1, 4]')
# args = parser.parse_args()
#
# # !!! Only for users who download our pre-trained checkpoint, comment the next four line if you are not !!!
# if os.path.basename(args.loadmodel) == 'DFF-DFV.tar':
#     args.use_diff = 1
# else:
#     args.use_diff = 0
#
# # dataloader
#
# # construct model
# model = DFFNet(clean=False, level=args.level, use_diff=args.use_diff)
# model = nn.DataParallel(model)
# model.cpu()
# ckpt_name = os.path.basename(os.path.dirname(args.loadmodel))
#
# if args.loadmodel is not None:
#     pretrained_dict = torch.load(
#         args.loadmodel, map_location=torch.device('cpu'))
#     pretrained_dict['state_dict'] = {
#         k: v for k, v in pretrained_dict['state_dict'].items() if 'disp' not in k}
#     model.load_state_dict(pretrained_dict['state_dict'], strict=False)
# else:
#     print('run with random init')
# print('Number of model parameters: {}'.format(
#     sum([p.data.nelement() for p in model.parameters()])))


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


def calmetrics(pred, target, mse_factor, accthrs, bumpinessclip=0.05, ignore_zero=True):
    metrics = np.zeros((1, 7 + len(accthrs)), dtype=float)

    if target.sum() == 0:
        return metrics

    pred_ = np.copy(pred)
    if ignore_zero:
        pred_[target == 0.0] = 0.0
        numPixels = (target > 0.0).sum()  # number of valid pixels
    else:
        numPixels = target.size

    # euclidean norm
    metrics[0, 0] = np.square(pred_ - target).sum() / numPixels * mse_factor

    # RMS
    metrics[0, 1] = np.sqrt(metrics[0, 0])

    # log RMS
    logrms = (np.ma.log(pred_) - np.ma.log(target))
    metrics[0, 2] = np.sqrt(np.square(logrms).sum() / numPixels)

    # absolute relative
    metrics[0, 3] = np.ma.divide(
        np.abs(pred_ - target), target).sum() / numPixels

    # square relative
    metrics[0, 4] = np.ma.divide(
        np.square(pred_ - target), target).sum() / numPixels

    # accuracies
    acc = np.ma.maximum(np.ma.divide(pred_, target),
                        np.ma.divide(target, pred_))
    for i, thr in enumerate(accthrs):
        metrics[0, 5 + i] = (acc < thr).sum() / numPixels * 100.

    # badpix
    metrics[0, 8] = (np.abs(pred_ - target) > 0.07).sum() / numPixels * 100.

    # bumpiness -- Frobenius norm of the Hessian matrix
    diff = np.asarray(pred - target, dtype='float64')  # PRED or PRED_
    chn = diff.shape[2] if len(diff.shape) > 2 else 1
    bumpiness = np.zeros_like(pred_).astype('float')
    for c in range(0, chn):
        if chn > 1:
            diff_ = diff[:, :, c]
        else:
            diff_ = diff
        dx = skf.scharr_v(diff_)
        dy = skf.scharr_h(diff_)
        dxx = skf.scharr_v(dx)
        dxy = skf.scharr_h(dx)
        dyy = skf.scharr_h(dy)
        dyx = skf.scharr_v(dy)
        hessiannorm = np.sqrt(
            np.square(dxx) + np.square(dxy) + np.square(dyy) + np.square(dyx))
        bumpiness += np.clip(hessiannorm, 0, bumpinessclip)
    bumpiness = bumpiness[target > 0].sum() if ignore_zero else bumpiness.sum()
    metrics[0, 9] = bumpiness / chn / numPixels * 100.

    return metrics


MIN_MSE = 1e9


def Eval(data_path, mdl_path, image_size=(383, 552), PROJ_DIR=None, epoch=None, stride=1, show_img=False):
    global MIN_MSE
    # logging.debug("data_path={}, mdl_path={}".format(data_path, mdl_path))
    imgs_and_loss_helper = ImgsAndLossHelper.ImgsAndLossHelper(
        PROJ_DIR + "saved_imgs/eval/", data_type="eval", stride=stride, show_img=show_img)

    model = torch.load(mdl_path)['net'].cuda()
    model.eval()

    # Calculate pad size for images
    test_pad_size = (
        np.ceil((image_size[0] / 32)) * 32, np.ceil((image_size[1] / 32)) * 32)
    batch_size = 1

    # Create test set transforms
    transform_test = [DDFF12Loader.ToTensor(),
                      DDFF12Loader.PadSamples(test_pad_size),
                      DDFF12Loader.Normalize(mean_input=[0.485, 0.456, 0.406], std_input=[0.229, 0.224, 0.225])]
    transform_test = torchvision.transforms.Compose(transform_test)

    test_set = DDFF12Loader(data_path, stack_key="stack_val", disp_key="disp_val",
                            transform=transform_test, n_stack=5, min_disp=0.02, max_disp=0.28, b_test=True)
    dataloader = DataLoader(test_set, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # metric prepare
    accthrs = [1.25, 1.25 ** 2, 1.25 ** 3]
    avgmetrics = np.zeros((1, 7 + len(accthrs) + 1), dtype=float)
    test_num = len(dataloader)
    time_rec = np.zeros(test_num)
    for inx, (img_stack, disp, foc_dist) in enumerate(dataloader):
        if inx % 10 == 0:
            logging.debug('processing: {}/{}'.format(inx, test_num))

        # if inx not in [4, 5, 6, 7]:
        #     continue

        img_stack = Variable(torch.FloatTensor(img_stack)).cuda()
        foc_dist = Variable(torch.FloatTensor(foc_dist)).cuda()
        gt = Variable(torch.FloatTensor(disp))

        with torch.no_grad():
            # torch.cuda.synchronize()
            start_time = time.time()
            diff_imgs, predss, stdss, costss = model(img_stack, foc_dist)
            # torch.cuda.synchronize()
            # print('time = %.2f' % (ttime*1000) )
            ttime = (time.time() - start_time)
            time_rec[inx] = ttime

        [pred_2s, pred_4s, pred_8s, pred_16s, pred_32s] = predss
        preds = [pred_2s[0], pred_4s[0], pred_8s[0], pred_16s[0], pred_32s[0]]
        [std_2s, std_4s, std_8s, std_16s, std_32s] = stdss
        stds = [std_2s[0], std_4s[0], std_8s[0], std_16s[0], std_32s[0]]

        # logging.debug("preds[0].shape={}".format(preds[0].shape))
        for b in range(batch_size):
            std = stds[0][b][0].squeeze().cpu().numpy()[
                :image_size[0], :image_size[1]]
            pred_disp = preds[0][b][0].squeeze().cpu().numpy()[
                :image_size[0], :image_size[1]]
            # logging.debug("pred_disp.shape={}".format(pred_disp.shape))
            # logging.debug("gt[b].shape={}".format(gt[b].shape))
            gt_disp = gt[b].squeeze().numpy()[:image_size[0], :image_size[1]]

            metrics = calmetrics(pred_disp, gt_disp, 1.0, accthrs,
                                 bumpinessclip=0.05, ignore_zero=True)
            avgmetrics[:, :-1] += metrics
            avgmetrics[:, -1] += std.mean()

            ########################## show imgs ##############################
            # img_stack = diff_imgs
            single_preds = torch.stack(
                (preds[0][0], preds[1][0], preds[2][0], preds[0][0] - preds[1][0], stds[0][0], stds[1][0]), dim=0)
            imgs_and_loss_helper.Update(
                img_stack[0], disp[0], single_preds, foc_dist[0], metrics[0, 0], epoch, inx, test_num)
            # logging.debug("id{} mse={}".format(inx, metrics[0, 0]))
            # cv2.waitKey(0)
            # SplitAndPred(PROJ_DIR, model, img_stack, foc_dist, disp, epoch, inx, test_num)
            ###################################################################

        # torch.cuda.empty_cache()

    final_res = (avgmetrics / (test_num*batch_size))[0]
    MIN_MSE = min(final_res[0], MIN_MSE)
    # remove badpix result, we do not use it in our paper
    final_res = np.delete(final_res, 8)
    final_log = ('==============  Final result =================') + '\n'
    final_log += ("\n  " + ("{:>10} | " * 10).format("MSE", "RMS", "log RMS",
                                                     "Abs_rel", "Sqr_rel", "a1", "a2", "a3", "bump", "avgUnc")) + '\n'
    final_log += (("  {: 2.6f}  " * 10).format(*final_res.tolist())) + '\n'
    # first one usually very large due to the pytorch warm up, discard
    final_log += ('runtime mean = {}'.format(np.mean(time_rec[1:]))) + '\n'
    final_log += ('MIN_MSE = {}'.format(MIN_MSE)) + '\n'
    logging.debug(final_log)


if __name__ == '__main__':
    ConfigLogging()

    logging.debug("Start eval")
    epoch = 233*3
    stride = 20  # save image once every 20 step
    show_img = False
    data_path = "../../data_sets/ddff-dfv/ddff-dataset-trainval.h5"
    # mdl_path = "../../saved_models/WDFV_e{:0>3d}.depth_net.mdl".format(epoch)
    mdl_path = "../../saved_models/WDFV-DDFF_431.depth_net.mdl"

    Eval(data_path, mdl_path, PROJ_DIR="../../",
         epoch=epoch, stride=stride, show_img=show_img)
