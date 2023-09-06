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


class ContextBlk(nn.Module):
    def __init__(self, in_c, ratio, pool_type="att", fusion_types=("add",)):
        super().__init__()

        assert pool_type in ["avg", "att"]
        assert all(f in ["add", "mul"] for f in fusion_types)
        assert len(fusion_types) > 0

        self.in_c = in_c
        self.ratio = ratio
        self.planes = int(self.in_c * self.ratio)
        self.pool_type = pool_type
        self.fusion_types = fusion_types

        if self.pool_type == "att":
            self.conv_mask = nn.Conv2d(self.in_c, 1, kernel_size=1)
            self.soft_max = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if "add" in self.fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.in_c, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.planes, self.in_c, kernel_size=1)
            )
        else:
            self.channel_add_conv = None

        if "mul" in self.fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.in_c, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.planes, self.in_c, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def SpatialPool(self, x):
        b, c, h, w = x.shape

        if self.pool_type == "att":
            input_x = x.view(b, c, h*w)
            # b, 1, c, h*w
            input_x = input_x.unsqueeze(dim=1)

            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(b, 1, h*w)
            context_mask = self.soft_max(context_mask)
            # b, 1, h*w, 1
            context_mask = context_mask.unsqueeze(-1)

            context = torch.matmul(input_x, context_mask)
            context = context.view(b, c, 1, 1)
        else:
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        context = self.SpatialPool(x)
        out = x

        if self.channel_mul_conv is not None:
            c_mul = torch.sigmoid(self.channel_mul_conv(context))
            out = out * c_mul

        if self.channel_add_conv is not None:
            c_add = self.channel_add_conv(context)
            out = out + c_add

        return out
