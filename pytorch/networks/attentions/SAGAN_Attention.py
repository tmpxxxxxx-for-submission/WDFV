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


class SAGAN_Attention(nn.Module):
    def __init__(self, in_c, mid_c=None):
        super().__init__()

        if mid_c is None:
            mid_c = in_c//8

        self.query_conv = nn.Conv2d(
            in_c, mid_c, kernel_size=1, stride=1, padding=0)
        self.key_conv = nn.Conv2d(
            in_c, mid_c, kernel_size=1, stride=1, padding=0)
        self.value_conv = nn.Conv2d(
            in_c, in_c, kernel_size=1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, w, h = x.size()
        # b x wh x c
        query = self.query_conv(x).view(b, -1, w*h).permute(0, 2, 1)

        # b x c x wh
        key = self.key_conv(x).view(b, -1, w*h)

        # transpose multiple
        energy = torch.bmm(query, key)
        # b x wh x wh
        attention = self.softmax(energy)

        # b x c x wh
        value = self.value_conv(x).view(b, -1, w*h)

        # apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(b, c, w, h)

        out = self.gamma*out + x
        return out, attention
