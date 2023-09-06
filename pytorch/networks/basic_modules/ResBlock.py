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
import sys
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
from ..attentions import CBAM


class ResBasicBlock2D(nn.Module):
    # 这里使用BatchNorm与InstanceNorm效果似乎差不多
    # 理论上来说InstanceNorm适合于风格迁移，那么应该也更适合SFF
    # 感觉上InstanceNorm更稳定
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, c_ratio=16, reverse_attention=False):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size,
                               stride, padding, bias=True)
        self.bn1 = nn.InstanceNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size,
                               1, padding, bias=True)
        self.bn2 = nn.InstanceNorm2d(out_c)

        if reverse_attention is not None:
            self.cbam = CBAM.CBAM(
                out_c, n_dim=2, c_ratio=c_ratio, reverse_attention=reverse_attention)
        else:
            self.cbam = None

        if stride > 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1,
                          stride=stride, bias=True),
                nn.InstanceNorm2d(out_c)
            )
        else:
            self.downsample = None

    def forward(self, x):
        ori_x = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.cbam is not None:
            x = self.cbam(x)

        if self.downsample is not None:
            ori_x = self.downsample(ori_x)

        x = x + ori_x
        return self.relu(x)


class ResBasicBlock3D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=(1, 1, 1), padding=1, c_ratio=16, reverse_attention=False):
        super().__init__()
        if type(stride) is int:
            stride = (stride, stride, stride)
        self.stride = stride

        self.conv1 = nn.Conv3d(in_c, out_c, 3, stride, 1, bias=True)
        self.bn1 = nn.InstanceNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_c, out_c, 3, (1, 1, 1), 1, bias=True)
        self.bn2 = nn.InstanceNorm3d(out_c)

        if reverse_attention is not None:
            self.cbam = CBAM.CBAM(
                out_c, n_dim=3, c_ratio=c_ratio, reverse_attention=reverse_attention)
        else:
            self.cbam = None

        if max(stride) > 1 or in_c != out_c:
            # TODO: 这里stride!=1可能会有问题
            self.downsample = nn.Sequential(
                nn.Conv3d(in_c, out_c, 1, stride, 0, bias=True),
                nn.InstanceNorm3d(out_c)
            )
        else:
            self.downsample = None

    def forward(self, x):
        ori_x = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.cbam is not None:
            x = self.cbam(x)

        if self.downsample is not None:
            ori_x = self.downsample(ori_x)

        return self.relu(x + ori_x)
