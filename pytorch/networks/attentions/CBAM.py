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
from torchviz import make_dot
from matplotlib import pyplot as plt
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tfs
from torch import nn
from torch.autograd import Variable
from datetime import datetime
from PIL import Image
from skimage import io as skiio


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_c, n_dim=2, ratio=16):
        super().__init__()
        if n_dim == 2:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)

            self.shared_MLP = nn.Sequential(
                nn.Conv2d(in_c, in_c // ratio, 1, bias=True),
                # hw为1，不能使用InstanceNorm；使用BatchNorm会使网络波动剧烈，无法收敛
                # LeakyReLU or ReLU 区别不大
                nn.LeakyReLU(True),
                nn.Conv2d(in_c // ratio, in_c, 1, bias=True),
                # hw为1，不能使用InstanceNorm；使用BatchNorm会使网络波动剧烈，无法收敛
            )
        elif n_dim == 3:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.max_pool = nn.AdaptiveMaxPool3d(1)

            self.shared_MLP = nn.Sequential(
                nn.Conv3d(in_c, in_c // ratio, 1, bias=True),
                # hw为1，不能使用InstanceNorm；使用BatchNorm会使网络波动剧烈，无法收敛
                # LeakyReLU or ReLU 区别不大
                nn.LeakyReLU(True),
                nn.Conv3d(in_c // ratio, in_c, 1, bias=True),
                # hw为1，不能使用InstanceNorm；使用BatchNorm会使网络波动剧烈，无法收敛
            )
        else:
            logging.error("ChannelAttentionModule: unsupport dimension")

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, n_dim=2):
        super().__init__()
        if n_dim == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, stride=1,
                          padding=3, bias=True),
                nn.InstanceNorm2d(1, affine=False),
            )
        elif n_dim == 3:
            self.conv = nn.Sequential(
                nn.Conv3d(2, 1, kernel_size=7, stride=1,
                          padding=3, bias=True),
                nn.InstanceNorm3d(1, affine=False),
            )

        if n_dim == 2:
            self.norm = nn.InstanceNorm2d(1, affine=False)
        elif n_dim == 3:
            self.norm = nn.InstanceNorm3d(1, affine=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # logging.debug("SAM, x.shape={}".format(x.shape))
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, sth = torch.max(x, dim=1, keepdim=True)
        # logging.debug("SAM, avg_out.shape={}, max_out.shape={}".format(avg_out.shape, max_out.shape))
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.norm(self.conv(out)))
        return out


class CBAM(nn.Module):
    def __init__(self, in_c, n_dim=2, c_ratio=16, reverse_attention=False):
        super().__init__()
        logging.debug("creating CBAM: params={}".format(locals()))
        self.reverse_attention = reverse_attention
        self.channel_attention = ChannelAttentionModule(
            in_c, n_dim=n_dim, ratio=c_ratio)
        self.spatial_attention = SpatialAttentionModule(n_dim=n_dim)
        logging.debug("finish creating CBAM")

    def forward(self, x):
        # x_ori = x.clone().detach()
        c_out = self.channel_attention(x)
        if self.reverse_attention:
            c_out = 1 - c_out
        # logging.debug("CBAM, c_out.shape={}".format(c_out.shape))
        x = c_out * x
        # x_after_c = x.clone().detach()
        s_out = self.spatial_attention(x)
        if self.reverse_attention:
            s_out = 1 - s_out
        # logging.debug("CBAM, s_out.shape={}".format(s_out.shape))
        x = s_out * x

        return x
        if s_out.shape[1] == 1 and s_out.shape[2] == s_out.shape[3]:
            # logging.debug("CBAM, x.shape={}".format(x.shape))
            tmp_x = x.cpu().detach().numpy()[0]
            s_out = s_out.squeeze(dim=1)
            s_out = s_out.cpu().detach().numpy()
            imgs = []
            x_imgs = []
            for idx in range(10):
                imgs += [cv2.resize(s_out[idx], (s_out[idx].shape[0] * 5,
                                                 s_out[idx].shape[1] * 5), interpolation=cv2.INTER_LINEAR)]
                x_imgs += [cv2.resize(tmp_x[idx], (tmp_x[idx].shape[0] * 5,
                                                   tmp_x[idx].shape[1] * 5), interpolation=cv2.INTER_LINEAR)]
                # imgs[-1] = (imgs[-1] - imgs[-1].min())/(imgs[-1].max() - imgs[-1].min())
                # x_imgs[-1] = (x_imgs[-1] - x_imgs[-1].min())/(x_imgs[-1].max() - x_imgs[-1].min())
                if tmp_x[idx].max() == tmp_x[idx].min():
                    logging.error("tmp_x[{}].max() == tmp_x[{}].min() = {}".format(
                        idx, idx, tmp_x[idx].max()))
            s_out = imgs
            upper_imgs = cv2.hconcat(
                [s_out[0], s_out[1], s_out[2], s_out[3], s_out[4]])
            lower_imgs = cv2.hconcat(
                [s_out[5], s_out[6], s_out[7], s_out[8], s_out[9]])
            upper_imgs2 = cv2.hconcat(
                [x_imgs[0], x_imgs[1], x_imgs[2], x_imgs[3], x_imgs[4]])
            lower_imgs2 = cv2.hconcat(
                [x_imgs[5], x_imgs[6], x_imgs[7], x_imgs[8], x_imgs[9]])
            imgs = cv2.vconcat(
                [upper_imgs, lower_imgs, upper_imgs2, lower_imgs2])
            # logging.debug(
            #    "CBAM, imgs.max={}, imgs.min={}".format(imgs.max(), imgs.min()))
            cv2.imshow("s_out_{}".format(x.shape[2]), imgs)
            cv2.waitKey(10)

        return x
