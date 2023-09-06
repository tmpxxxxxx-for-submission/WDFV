#!/home/z_t_h_/Workspace/libs/anaconda3/bin/python3.9
# -*- coding: UTF-8 -*-

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
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tfs
from torch import nn
from torch import optim
from torch.autograd import Variable
from datetime import datetime
from PIL import Image
from skimage import io as skiio


class ImgsAndLossHelper:
    def __init__(self, save_folder=None, stride=5, data_type="train", show_img=True):
        self.save_folder = save_folder
        self.data_type = data_type
        self.show_img = show_img

        self.all_loss = []
        self.showing_loss = [0]
        self.stride = stride
        self.update_cnt = 0
        self.max_showing_loss = 200
        self.init_time = datetime.now()
        self.last_save_time = self.init_time

    def Update(self, imgs, label, all_res, focal_dist, loss, epoch, batch_idx, data_len):

        # logging.debug("ImgsAndLossHelper::Update imgs.shape={}, label.shape={}. all_res.shape={}".format(
        #    imgs.shape, label.shape, all_res.shape))

        if torch.is_tensor(loss):
            loss = loss.detach().cpu().numpy()

        self.all_loss.append(loss)
        self.showing_loss[-1] += loss
        self.update_cnt += 1

        if self.update_cnt % self.stride == 0:
            pre_time = self.last_save_time
            cur_time = datetime.now()
            logging.info("{}... {:0>5d}/{:0>5d}-{:0>3d}, loss: {:.6f}, this epoch time: {}, total time: {}\n".format(self.data_type,
                                                                                                                     batch_idx, data_len, epoch, loss, cur_time - pre_time, cur_time - self.init_time))
            self.last_save_time = cur_time
            self.showing_loss[-1] /= self.stride
            # cv image shape: (W, H, C)
            imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
            label = label.cpu().permute(1, 2, 0).numpy()
            all_res = all_res.cpu().detach().permute(0, 2, 3, 1).numpy()
            focal_dist = focal_dist.cpu().numpy()

            # loss_mat = self.DrawLossMat(loss)
            imgs_mat = self.DrawImgsMat(imgs, label, all_res, focal_dist)

            if self.show_img:
                cv2.imshow("{}_imgs".format(self.data_type), imgs_mat)
                # cv2.imshow("{}_loss".format(self.data_type), loss_mat)
                cv2.waitKey(1)

            if self.save_folder is not None:
                imgs_filename = "{}_imgs_e{:0>3d}c{:0>5d}%{:0>5d}.png".format(
                    self.data_type, epoch, batch_idx, data_len)
                loss_filename = "{}_loss_e{:0>3d}c{:0>5d}%{:0>5d}.png".format(
                    self.data_type, epoch, batch_idx, data_len)
                imgs_filename = str(os.path.join(
                    self.save_folder, imgs_filename))
                loss_filename = str(os.path.join(
                    self.save_folder, loss_filename))
                # logging.debug("imgs_filename={}".format(imgs_filename))
                cv2.imwrite(imgs_filename, imgs_mat)
                # cv2.imwrite(loss_filename, loss_mat)

            self.showing_loss.append(0)
            if len(self.showing_loss) > self.max_showing_loss:
                self.showing_loss = self.showing_loss[1:]

    def DrawLossMat(self, loss):
        min_loss = min(self.showing_loss)
        max_loss = max(max(self.showing_loss), min_loss + 1e-8)
        # logging.debug("max_loss:{}, min_loss:{}".format(max_loss, min_loss))
        hist_H = 300
        hist_W = 1410
        hist_img = np.zeros([hist_H + 3, hist_W + 3, 3], np.uint8)
        hist_img += 255
        hist_x = 0
        hist_y = 0
        pre_hist_y = -1
        for idx in range(len(self.showing_loss)):
            hist_x += 7
            hist_y = int((self.showing_loss[idx] - min_loss) /
                         (max_loss - min_loss) * hist_H + 1)
            hist_y = 300 - hist_y  # (0,0)在左上角
            # logging.debug("hist_x:{}, hist_y:{}".format(hist_x, hist_y))
            cv2.circle(hist_img, (hist_x, hist_y), 5, (0, 0, 238), 1)
            if idx >= 1:
                cv2.line(hist_img, (hist_x - 7, pre_hist_y),
                         (hist_x, hist_y), (0xFF, 0xCC, 0x66), 2)
            pre_hist_y = hist_y

        cv2.putText(hist_img, str(max_loss), (0, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.433, (255, 112, 132), 1)
        cv2.putText(hist_img, str(min_loss), (0, 290),
                    cv2.FONT_HERSHEY_DUPLEX, 0.433, (255, 112, 132), 1)

        return hist_img

    def Normalize(self, img, min_val, max_val):
        img = (img - min_val) / (max_val - min_val)
        img = (img * 255).astype(np.uint8)
        return img

    def Convert2JetMap(self, img, min_val, max_val):
        img = self.Normalize(img, min_val, max_val)

        if img.shape[2] == 1:
            img = img.squeeze(axis=2)

        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        return img

    def DrawImgsMat(self, imgs, label, all_res, focal_dist):
        label_max = focal_dist[0]
        label_min = focal_dist[-1]
        gaussian = tfs.GaussianBlur(7)

        # logging.debug("in show imgs, imgs.shape:{}, label.shape:{}, all_res.shape:{}".format(
        #    imgs.shape, label.shape, all_res.shape))
        # logging.debug("label_min={}, label_max={}, label.min={}, label.max={}".format(label_min, label_max, label.min(), label.max()))

        mask = (label > label_min) & (label < label_max)
        res_diff = label.copy()
        res_diff[mask] = np.absolute(label[mask] - all_res[0][mask])
        label_edge = torch.from_numpy(label).squeeze().unsqueeze(0)
        label_edge = label - gaussian(label_edge).permute(1, 2, 0).numpy()

        imgs = self.Normalize(imgs, imgs.min(), imgs.max())
        label = self.Convert2JetMap(label, label_min, label_max)
        label_edge = self.Convert2JetMap(label_edge, label_min, label_max)
        res_diff = self.Convert2JetMap(
            res_diff, res_diff.min(), res_diff.max())
        if len(all_res) == 3:
            all_res = self.Convert2JetMap(all_res, label_min, label_max)
        elif len(all_res) == 4 or len(all_res) == 6:
            tmp_all_res = []
            for idx in range(all_res.shape[0]):
                tmp_all_res.append(self.Convert2JetMap(
                    all_res[idx], label_min, label_max))
            all_res = tmp_all_res
        else:
            logging.error(
                "wrong all_res.shape!!! which is {}".format(all_res.shape))
        #logging.debug("jet map all_res[0].shape{}".format(all_res[0].shape))
        #logging.debug("jet map res_diff.shape{}".format(res_diff.shape))

        img_diffs = []
        for idx in range(2, imgs.shape[0]):
            img_diff = imgs[idx] - imgs[idx-2]
            img_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)
            img_diff = cv2.GaussianBlur(img_diff, (5, 5), 0)
            img_diff = cv2.merge((img_diff, img_diff, img_diff))
            img_diffs.append(img_diff)

        arranged_imgs = cv2.hconcat(
            [imgs[0], imgs[1], imgs[2], imgs[3], imgs[4]])
        res_imgs = cv2.hconcat(
            [label, all_res[0], all_res[1], all_res[2], all_res[3]])
        extra_res_imgs = cv2.hconcat(
            [res_diff, label_edge, img_diffs[2], all_res[4], all_res[5]]
        )
        # other_imgs = cv2.hconcat([img_diff, res_diff, label, all_res])
        all_imgs = cv2.vconcat([arranged_imgs, res_imgs, extra_res_imgs])

        return all_imgs

    def Close(self):
        # TODO write all_loss to file
        pass
