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
from loss_func import SSIM


class ProjectIndexes(nn.Module):
    def __init__(self, batch_size, h, w):
        super().__init__()

        self.batch_size = batch_size
        self.h = h
        self.w = w

        meshgrid = np.meshgrid(range(self.h), range(self.w), indexing="xy")
        # axis=0未新增维度
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(
            self.id_coords), requires_grad=False)

        # z && homogeneous coordinate
        self.ones = nn.Parameter(torch.ones(
            self.batch_size, 1, self.h * self.w), requires_grad=False)

        # pix_coords = (x,y)
        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], dim=0), dim=0)
        self.pix_coords = self.pix_coords.repeat(self.batch_size, 1, 1)
        # (x,y,1,1)
        self.pix_coords = nn.Parameter(
            torch.cat([self.pix_coords, self.ones, self.ones], dim=1), requires_grad=False)

    def forward(self, P):
        # for idx in range(1, 300, 10):
        #     logging.debug("self.pix_coords[{}]={}".format(
        #         idx, self.pix_coords[0, :, idx]))
        proj_pts = torch.matmul(P, self.pix_coords)
        return proj_pts


class AffineParamToMatrix(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, angles, transforms, h, w, invert=False):
        R = self.AngleToMatrix(angles)
        T = transforms.clone()

        if invert:
            R = R.transpose(1, 2)
            T *= -1

        T = self.TransformsToMatrix(T, h, w)

        if invert:
            M = torch.matmul(R, T)
        else:
            M = torch.matmul(T, R)

        return M

    def TransformsToMatrix(self, translation_vector, h, w):
        """
        Convert a translation vector into a 4x4 transformation matrix
        """
        T = torch.zeros(translation_vector.shape[0], 4, 4).to(
            device=translation_vector.device)

        t = translation_vector.contiguous().view(-1, 3, 1)

        T[:, 0, 0] = 1
        T[:, 1, 1] = 1
        T[:, 2, 2] = 1
        T[:, 3, 3] = 1
        T[:, :3, 3, None] = t

        T[:, 0, 3] = T[:, 0, 3] * h
        T[:, 1, 3] = T[:, 1, 3] * w

        return T

    def AngleToMatrix(self, vec):
        """Convert an axisangle rotation into a 4x4 transformation matrix
            (adapted from https://github.com/Wallacoloo/printipi)
            Input 'vec' has to be Bx1x3
        """
        angle = torch.norm(vec, 2, 2, True)
        axis = vec / (angle + 1e-7)

        ca = torch.cos(angle)
        sa = torch.sin(angle)
        C = 1 - ca

        x = axis[..., 0].unsqueeze(1)
        y = axis[..., 1].unsqueeze(1)
        z = axis[..., 2].unsqueeze(1)

        xs = x * sa
        ys = y * sa
        zs = z * sa
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC

        rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

        rot[:, 0, 0] = torch.squeeze(x * xC + ca)
        rot[:, 0, 1] = torch.squeeze(xyC - zs)
        rot[:, 0, 2] = torch.squeeze(zxC + ys)
        rot[:, 1, 0] = torch.squeeze(xyC + zs)
        rot[:, 1, 1] = torch.squeeze(y * yC + ca)
        rot[:, 1, 2] = torch.squeeze(yzC - xs)
        rot[:, 2, 0] = torch.squeeze(zxC - ys)
        rot[:, 2, 1] = torch.squeeze(yzC + xs)
        rot[:, 2, 2] = torch.squeeze(z * zC + ca)
        rot[:, 3, 3] = 1

        tmp_rot = torch.zeros(vec.shape[0], 4, 4).to(device=vec.device)
        tmp_rot[:, 0, 0] = 1
        tmp_rot[:, 1, 1] = 1
        tmp_rot[:, 2, 2] = 1
        tmp_rot[:, 3, 3] = 1

        return tmp_rot


class ReprojectionError(nn.Module):
    def __init__(self, batch_size, h, w):
        super().__init__()
        self.batch_size = batch_size
        self.h = h
        self.w = w

        self.proj_idxes_func = ProjectIndexes(batch_size, h, w)
        self.affine_param_to_matrix_func = AffineParamToMatrix()
        self.eps = 1e-9

        self.ssim_criterion = SSIM.SSIM(
            CUDA_ID=0, val_range=1.0, kernel_size=7)
        self.l1_criterion = torch.nn.SmoothL1Loss(reduction="mean")
        self.rgb2gray = tfs.functional.rgb_to_grayscale

    def ShowReprojImgs(self, affined_img, corrected_affined_img, ori_img):
        affined_img = affined_img[0].permute(1, 2, 0)
        corrected_affined_img = corrected_affined_img[0].permute(1, 2, 0)
        ori_img = ori_img[0].permute(1, 2, 0)
        diff_img = torch.abs(ori_img - corrected_affined_img)

        show_img = torch.cat(
            [affined_img, corrected_affined_img, ori_img, diff_img], dim=1)
        show_img = show_img.detach().cpu().numpy()
        cv2.imshow("repojection images", show_img)
        # cv2.waitKey(0)

    def forward(self, angles, transforms, affined_img, ori_img):
        # logging.debug("angles.shape={}, transforms.shape={}".format(
        #     angles.shape, transforms.shape))
        # logging.debug("angles={}, transforms={}".format(angles, transforms))
        P = self.affine_param_to_matrix_func(
            angles, transforms, self.h, self.w)
        # logging.debug("P.shape={}".format(P.shape))
        # logging.debug("P=\n{}".format(P))

        proj_idxes = self.proj_idxes_func(P)
        # z -> 1
        # proj_idxes = proj_idxes[:, :2, :] / \
        #     (proj_idxes[:, 2, :].unsqueeze(1) + self.eps)
        proj_idxes = proj_idxes[:, :2, :]
        proj_idxes = proj_idxes.view(self.batch_size, 2, self.h, self.w)
        proj_idxes = proj_idxes.permute(0, 2, 3, 1)
        proj_idxes[..., 0] /= self.w - 1
        proj_idxes[..., 1] /= self.h - 1
        proj_idxes = (proj_idxes - 0.5) * 2

        corrected_affined_img = F.grid_sample(
            affined_img, proj_idxes, padding_mode="zeros")

        affined_img = self.rgb2gray(affined_img)
        corrected_affined_img = self.rgb2gray(corrected_affined_img)
        ori_img = self.rgb2gray(ori_img)

        # logging.debug("affined_img.shape={}, proj_idxes.shape={}".format(
        #     affined_img.shape, proj_idxes.shape))

        ssim_loss = self.ssim_criterion(corrected_affined_img, ori_img)
        l1_loss = self.l1_criterion(corrected_affined_img, ori_img)

        # logging.debug("cur l1_loss={}".format(l1_loss))
        self.ShowReprojImgs(affined_img.clone(),
                            corrected_affined_img.clone(), ori_img.clone())

        # return 0.85*ssim_loss + 0.15*l1_loss
        return l1_loss
