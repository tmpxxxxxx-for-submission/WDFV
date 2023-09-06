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


class GateAttention(nn.Module):
    def __init__(self, in_up_x_c, in_ori_x_c, mid_c, dim=2):
        super().__init__()

        assert dim == 2 or dim == 3
        if dim == 2:
            conv = nn.Conv2d
            norm = nn.InstanceNorm2d
        elif dim == 3:
            conv = nn.Conv3d
            norm = nn.InstanceNorm3d

        self.W_up_x = nn.Sequential(
            conv(in_up_x_c, mid_c, kernel_size=1,
                 stride=1, padding=0, bias=True),
            norm(mid_c, affine=False),
        )
        self.W_ori_x = nn.Sequential(
            conv(in_ori_x_c, mid_c, kernel_size=1,
                 stride=1, padding=0, bias=True),
            norm(mid_c, affine=False),
        )
        self.psi = nn.Sequential(
            conv(mid_c, 1, kernel_size=1, stride=1, padding=0, bias=True),
            norm(1, affine=False),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(True)

    def forward(self, up_x, ori_x):
        w_up_x = self.W_up_x(up_x)
        w_ori_x = self.W_ori_x(ori_x)
        psi = self.psi(self.relu(w_up_x + w_ori_x))

        return ori_x * psi
