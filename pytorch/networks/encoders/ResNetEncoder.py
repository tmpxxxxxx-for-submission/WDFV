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
from networks.basic_modules import BasicBlock, ResBlock


class ResNetEncoder2D(nn.Module):
    def __init__(self, pretrained=False, num_input_imgs=1):
        super().__init__()
        logging.debug("creating ResNetEncoder2D: params={}".format(locals()))

        self.conv1 = nn.Conv2d(num_input_imgs * 3, 64,
                               kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.InstanceNorm2d(64)
        self.relu = nn.LeakyReLU(True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.ResLayer2D(
            64, 64, stride=1, block_num=2, reverse_attention=None)
        self.layer2 = self.ResLayer2D(
            64, 128, stride=2, block_num=2, reverse_attention=False)
        self.layer3 = self.ResLayer2D(
            128, 256, stride=2, block_num=2, reverse_attention=False)
        self.layer4 = self.ResLayer2D(
            256, 512, stride=2, block_num=2, reverse_attention=False)

        logging.debug(
            "ResNetEncoder2D: finish creating ResNetEncoder2D")

    def ResLayer2D(self, in_c, out_c, stride, block_num, reverse_attention=False):
        blks = [ResBlock.ResBasicBlock2D(in_c, out_c, stride=stride,
                                         reverse_attention=None)]
        for cnt in range(1, block_num-1):
            blks += [ResBlock.ResBasicBlock2D(out_c, out_c, stride=1,
                                              reverse_attention=None)]
        blks += [ResBlock.ResBasicBlock2D(out_c, out_c, stride=1,
                                          reverse_attention=reverse_attention)]

        return nn.Sequential(*blks)

    def forward(self, x):
        # b*n,3,h,w

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
