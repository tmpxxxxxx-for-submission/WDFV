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


# 这个应该是MSSIM
class SSIM(nn.Module):
    def __init__(self, CUDA_ID, kernel_size=11, sigma=None, size_avg=True):
        super().__init__()
        if sigma is None:
            sigma = 1 if kernel_size == 1 else (kernel_size - 1) * 2.0
        logging.info("Init SSIM with params={},".format(locals()))

        self.size_avg = size_avg
        self.pad = kernel_size // 2
        self.kernel_size = kernel_size

        self.kernel = self.CreateGuassianKernel(kernel_size, sigma)
        if CUDA_ID >= 0:
            self.kernel = self.kernel.cuda(CUDA_ID)
        else:
            self.kernel = self.kernel.cpu()

    def CreateGuassianKernel(self, kernel_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - kernel_size//2)**2/(2*sigma**2)) for x in range(kernel_size)])
        guass = (gauss/gauss.sum()).unsqueeze(1)

        kernel = guass.mm(guass.t()).float().unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(1, 1, kernel_size, kernel_size).contiguous()
        return kernel

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

    def forward(self, img1, img2, valid_mask=None):
        # logging.debug("img1.max={}, img1.min={}, img2.max={}, img2.min={}".format(
        #     img1.max(), img1.min(), img2.max(), img2.min()))
        if valid_mask is not None:
            valid_mask = self.ErodeValidMask(valid_mask)
        else:
            valid_mask = torch.ones_like(img1).bool()

        # if valid_mask is None:
        #     valid_mask = torch.ones_like(img1).float()
        # else:
        #     valid_mask = valid_mask.float()
        # img1 = img1 * valid_mask
        # img2 = img2 * valid_mask

        val_range = max(img1.max(), img2.max()) - \
            min(img1.min(), img2.min()) + 1e-5

        # logging.debug("img1.shape={}, img2.shape={}, kernel.shape={}".format(
        #    img1.shape, img2.shape, self.kernel.shape))

        conv2d = nn.functional.conv2d
        mu1 = conv2d(img1, self.kernel, padding=self.pad, groups=1)
        mu2 = conv2d(img2, self.kernel, padding=self.pad, groups=1)

        mu1_sq = mu1.pow(2.0)
        mu2_sq = mu2.pow(2.0)
        mu1mu2 = mu1*mu2

        # sigma1_sq = conv2d(img1 * img1, self.kernel,
        #                   padding=1, groups=1) - mu1_sq
        # sigma2_sq = conv2d(img2 * img2, self.kernel,
        #                   padding=1, groups=1) - mu2_sq
        # sigma1sigma2 = conv2d(img1 * img2, self.kernel,
        #                      padding=1, groups=1) - mu1mu2
        sigma1_sq = conv2d((img1 - mu1) * (img1 - mu1),
                           self.kernel, padding=self.pad, groups=1)
        sigma2_sq = conv2d((img2 - mu2) * (img2 - mu2),
                           self.kernel, padding=self.pad, groups=1)
        sigma1sigma2 = conv2d((img1 - mu1) * (img2 - mu2),
                              self.kernel, padding=self.pad, groups=1)

        mu1_sq = mu1_sq + 1
        mu2_sq = mu2_sq + 1
        mu1mu2 = mu1mu2 + 1
        sigma1_sq = sigma1_sq + 1
        sigma2_sq = sigma2_sq + 1
        sigma1sigma2 = sigma1sigma2 + 1
        
        C1 = (0.01 * val_range) ** 2
        C2 = (0.03 * val_range) ** 2

        v1 = (2.0 * sigma1sigma2) + C2
        v2 = (sigma1_sq + sigma2_sq) + C2
        structure = v1/v2  # contrast sensitivity
        brightness = ((2 * mu1mu2) ** 2 + C1) / \
            ((mu1_sq + mu2_sq)**2 + C1)

        # ssim_map = ((2 * mu1mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
        ssim_map = structure[valid_mask] * brightness[valid_mask]
        # logging.debug("ssim={}, struct={}, bright={}, struct*bright={}".format(
        #     ssim_map.mean(), structure.mean(), brightness.mean(), (structure*brightness).mean()))
        ssim_loss = 1 - ssim_map.mean()

        # logging.debug("ssim_map={}, cs={}".format(ssim_map, cs))
        return ssim_loss  # , cs
