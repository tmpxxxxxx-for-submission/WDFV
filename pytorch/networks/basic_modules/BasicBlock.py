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


class Conv2dBnReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True, norm_type="IN"):
        super().__init__()
        self.norm_type = norm_type

        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        if norm_type is None:
            self.bn = None
        elif norm_type == "IN":
            self.bn = nn.InstanceNorm2d(out_c)
        elif norm_type == "BN":
            self.bn = nn.BatchNorm2d(out_c)
        else:
            assert False, "invalid norm_type {}".format(norm_type)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        (h, w) = (x.shape[-2], x.shape[-1])
        if self.bn is not None:
            if self.norm_type != "IN" or h+w>2:
                x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv3dBnReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True, relu=True, norm_type="IN"):
        super().__init__()
        self.norm_type = norm_type

        self.conv = nn.Conv3d(in_c, out_c, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        if norm_type is None:
            self.bn = None
        elif norm_type == "IN":
            self.bn = nn.InstanceNorm3d(out_c)
        elif norm_type == "BN":
            self.bn = nn.BatchNorm3d(out_c)
        else:
            assert False, "invalid norm_type {}".format(norm_type)

        if relu:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        (d, h, w) = (x.shape[-3], x.shape[-2], x.shape[-1])
        if self.bn is not None:
            if self.norm_type != "IN" or d+h+w>3:
                x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
