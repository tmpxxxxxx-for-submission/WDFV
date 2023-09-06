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
from tqdm import tqdm, trange
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tfs
from torch import nn
from torch import optim
from torch.autograd import Variable
from datetime import datetime
from PIL import Image
from skimage import io as skiio
from data_loaders import DFVData
from networks import IDFV
from tools import ImgsAndLossHelper


# 这个应该是MSSIM
class Laplacian(nn.Module):
    def __init__(self, CUDA_ID):
        super().__init__()
        lapla_kernel = np.array([
            [0., 1., 0.],
            [1., -4., 1.],
            [0., 1., 0.]
        ])

        lapla_kernel = torch.from_numpy(lapla_kernel).float().cuda(CUDA_ID)
        # (in_c, out_c, kernel_h, kernel_w)
        self.lapla_kernel = lapla_kernel.unsqueeze(0).unsqueeze(0)
        self.kernel_size = 3

    def ErodeValidMask(self, valid_mask):
        # logging.debug("valid_mask.shape={}, requires_grad={}".format(valid_mask.shape, valid_mask.requires_grad))
        new_valid_mask = []
        b, c, h, w = valid_mask.shape
        for b_idx in range(b):
            cur_mask = valid_mask[b_idx].squeeze()
            cur_mask = cur_mask.cpu().numpy().astype(np.uint8)
            cur_mask *= 255

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
            cur_mask = cv2.erode(cur_mask, kernel, iterations=1)

            cur_mask = torch.from_numpy(cur_mask).unsqueeze(0).bool()
            cur_mask = cur_mask.to(device=valid_mask.device)
            new_valid_mask.append(cur_mask)
        new_valid_mask = torch.stack(new_valid_mask, dim=0)
        # logging.debug("new_valid_mask.shape={}, requires_grad={}".format(new_valid_mask.shape, new_valid_mask.requires_grad))
        return new_valid_mask

    def forward(self, label, pred, valid_mask=None):
        if valid_mask is not None:
            valid_mask = self.ErodeValidMask(valid_mask)
        else:
            valid_mask = torch.ones_like(label).bool()

        conv2d = nn.functional.conv2d
        label_res = conv2d(label, self.lapla_kernel, stride=1, padding=1)
        pred_res = conv2d(pred, self.lapla_kernel, stride=1, padding=1)

        res = (label_res - pred_res) ** 2
        res = res[valid_mask].mean(0)

        return res
