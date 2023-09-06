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
from .basic_modules import BasicBlock, ResBlock
from .encoders import UNet3PlusEncoder, ResNetEncoder


class DecoderBlock(nn.Module):
    def __init__(self, level):
        super().__init__()

        self.classify_c = 80

        # 邻近输出的层用reverse_attention结果似乎会平滑一些
        # conv_32/16/8/4/2的out_c设为64的时候，效果会下降
        # 测试结果：无论是否模仿flowNet把classify结果cat近来，MSE最后都是1.27e-4，差不多
        # 测试结果：在conv_32/conv_16等之后使用spp，MSE会增大
        # 测试结果：这里使用Fuse，MSE会增大
        if level >= 4:
            # self.conv_32 = self.FuseFeatBlk(512, 128)
            # self.spp_32 = self.CreateSPPConv(256, 256)
            self.classify_32 = self.ClassifyBlk(128)
            self.up_classify_32 = self.UpsampleBlk(1, 1)

        if level >= 3:
            # self.conv_16 = self.FuseFeatBlk(160, self.classify_c)
            # self.spp_16 = self.CreateSPPConv(128, 128)
            self.classify_16 = self.ClassifyBlk(self.classify_c)
            self.up_classify_16 = self.UpsampleBlk(1, 1)

        if level >= 2:
            # self.conv_8 = self.FuseFeatBlk(160, self.classify_c)
            # self.spp_8 = self.CreateSPPConv(64, 64)
            self.classify_8 = self.ClassifyBlk(self.classify_c)
            self.up_classify_8 = self.UpsampleBlk(1, 1)

        if level >= 1:
            # self.conv_4 = self.FuseFeatBlk(160, self.classify_c)
            self.classify_4 = self.ClassifyBlk(self.classify_c)
            self.up_classify_4 = self.UpsampleBlk(1, 1)

            # self.conv_2 = self.FuseFeatBlk(160, self.classify_c)
            self.classify_2 = self.ClassifyBlk(self.classify_c)

    def FuseFeatBlk(self, in_c, out_c):
        return nn.Sequential(
            ResBlock.ResBasicBlock3D(
                in_c, out_c, stride=1, reverse_attention=False),
            ResBlock.ResBasicBlock3D(
                out_c, out_c, stride=1, reverse_attention=False),
        )

    def ClassifyBlk(self, out_c):
        return nn.Sequential(
            nn.Conv3d(out_c, out_c, 3, 1, 1, bias=True),
            # nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_c, 1, 3, 1, 1, bias=True),
            # nn.InstanceNorm3d(1),
        )

    def UpsampleBlk(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear"),
            BasicBlock.Conv3dBnReLU(in_c=in_c, out_c=out_c),
        )

    def CreateSPPConv(self, in_c, out_c):
        # TODO: IN / BN 哪个更合适？
        return torch.nn.ModuleList([
            BasicBlock.Conv3dBnReLU(
                in_c, out_c, kernel_size=1, stride=1, padding=0, norm_type="IN"),
            BasicBlock.Conv3dBnReLU(
                in_c, out_c, kernel_size=1, stride=1, padding=0, norm_type="IN"),
            BasicBlock.Conv3dBnReLU(
                in_c, out_c, kernel_size=1, stride=1, padding=0, norm_type="IN"),
            BasicBlock.Conv3dBnReLU(
                in_c, out_c, kernel_size=1, stride=1, padding=0, norm_type="IN")
        ])

    def SPP(self, x, pool_convs):
        _, _, d, h, w = x.shape
        res = x
        for i, pool_size in enumerate(np.linspace(1, min(d, h, w)//2, 4, dtype=int)):
            kernel_size = (int(d/pool_size),
                           int(h/pool_size), int(w/pool_size))
            out = F.avg_pool3d(x, kernel_size, stride=kernel_size)
            out = pool_convs[i](out)
            out = F.upsample(out, size=(d, h, w), mode='trilinear')
            res = res + 0.25*out
        res = F.leaky_relu(res/2., inplace=True)

        return res

    def forward(self, feat_2, feat_4=None, feat_8=None, feat_16=None, feat_32=None):
        if feat_32 is not None:
            # feat_32 = self.conv_32(feat_32)
            # feat_32 = self.SPP(feat_32, self.spp_32)
            cost_32 = self.classify_32(feat_32)
            up_cost_32_to_16 = self.up_classify_32(cost_32)

            cost_32 = cost_32.squeeze(1)
        else:
            cost_32 = None

        if feat_16 is not None:
            # feat_16 = self.conv_16(feat_16)
            # feat_16 = self.SPP(feat_16, self.spp_16)
            # feat_16 = torch.cat([feat_16, up_cost_32_to_16], dim=1)
            cost_16 = self.classify_16(feat_16)
            up_cost_16_to_8 = self.up_classify_16(cost_16)

            cost_16 = cost_16.squeeze(1)
        else:
            cost_16 = None

        if feat_8 is not None:
            # feat_8 = self.conv_8(feat_8)
            # feat_8 = self.SPP(feat_8,self.spp_8)
            # feat_8 = torch.cat([feat_8, up_cost_16_to_8], dim=1)
            cost_8 = self.classify_8(feat_8)
            up_cost_8_to_4 = self.up_classify_8(cost_8)

            cost_8 = cost_8.squeeze(1)
        else:
            cost_8 = None

        if feat_4 is not None:
            # feat_4 = self.conv_4(feat_4)

            # feat_4 = torch.cat([feat_4, up_cost_8_to_4], dim=1)
            cost_4 = self.classify_4(feat_4)
            up_cost_4_to_2 = self.up_classify_4(cost_4)

            cost_4 = cost_4.squeeze(1)
        else:
            cost_4 = None

        if feat_2 is not None:
            # feat_2 = self.conv_2(feat_2)

            # feat_2 = torch.cat([feat_2, up_cost_4_to_2], dim=1)
            cost_2 = self.classify_2(feat_2).squeeze(1)
        else:
            cost_2 = None

        return [cost_2, cost_4, cost_8, cost_16, cost_32]


class DisparityRegression(nn.Module):
    def __init__(self, divisor=1):
        super().__init__()
        self.divisor = divisor

    def forward(self, x, focal_dists, uncertainty=False):
        # logging.debug("DisparityRegression, x.shape={},focal_dists.shape={}".format(
        #    x.shape, focal_dists.shape))
        # b,n,1,1 <==> b,n,h,w
        focal_dists = focal_dists.unsqueeze(-1).unsqueeze(-1)
        # sum(概率*距离)=期望=预测的距离
        out = torch.sum(x * focal_dists, 1, keepdim=True) * self.divisor
        # logging.debug("DisparityRegression, out.shape={}".format(out.shape))

        # 偏离程度，用于求置信度
        std = torch.sqrt(
            torch.sum(x * (out - focal_dists)**2, 1, keepdim=True))
        return out, std.detach()


class WDFVNet(nn.Module):
    def __init__(self, level, use_diff):
        super().__init__()
        logging.debug("creating WDFVNet")

        self.level = level
        self.use_diff = use_diff

        # self.encoder2d = UNet3PlusEncoder.Unet3PlusEncoder2D(level)
        self.encoder2d = None
        self.encoder3d = UNet3PlusEncoder.Unet3PlusEncoder3D(level)

        # 默认level==4
        self.decoder_l4 = DecoderBlock(level=4) if level >= 4 else None

        self.disp_reg = DisparityRegression()
        self.soft_max = nn.Softmax(dim=1)

        for name, m in self.named_modules():
            if "feature_extraction" in name:
                # 预训练模型不初始化
                continue
            # logging.debug("------- cur name:{}".format(name))
            if isinstance(m, nn.Linear):
                # logging.debug("cur m:{}".format(m))
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                # logging.debug("cur m:{}".format(m))
                m.weight = nn.init.kaiming_normal_(
                    m.weight, a=1e-2)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

        logging.debug("finish creating DFV")

    def CreateDFV(self, focus_volume):
        diff = focus_volume[:, :, :-1] - focus_volume[:, :, 1:]
        return torch.cat([diff, focus_volume[:, :, -1:]], dim=2)

    def Feat2DPostProcess(self, b, n, h, w, feat_2ds, use_diff=False):
        feats_res = []
        for idx in range(len(feat_2ds)):
            feat_2d = feat_2ds[idx]
            # feat_3d = feat_3ds[idx]

            if feat_2d is None:
                feats_res = feats_res + [None]
                continue

            feat_2d = feat_2d.view(b, n, -1, h, w).permute(0, 2, 1, 3, 4)
            if use_diff:
                diff = feat_2d[:, :, 1:, :, :] - feat_2d[:, :, :-1, :, :]
                feat_2d = torch.cat([diff, feat_2d[:, :, -1:, :, :]], dim=2)
            feats_res = feats_res + [feat_2d]

        return feats_res

    def UpAndReg(self, costs, focal_dists, up_dims, up_2d_func, up_3d_func, soft_max_func):
        assert len(costs) == len(up_dims)
        preds, stds, res_costs = [], [], []
        for idx, cost in enumerate(costs):
            pred, std = None, None

            if cost is not None:
                if up_dims[idx] == 2:
                    cost = up_2d_func(cost)
                elif up_dims[idx] == 3:
                    cost = up_3d_func(cost.unsqueeze(1)).squeeze(1)
                else:
                    assert False

                cost = soft_max_func(cost)
                pred, std = self.disp_reg(cost, focal_dists, uncertainty=True)

            preds.append(pred)
            stds.append(std)
            res_costs.append(cost)

        return preds, stds, res_costs

    def Use2DEncoder(self, x):
        b, n, c, h, w = x.shape

        feat_2d_2s, feat_2d_4s, feat_2d_8s, feat_2d_16s, feat_2d_32s = self.encoder2d(
            x.view(b*n, c, h, w))

        if self.level >= 1:
            feat_2d_2s = self.Feat2DPostProcess(
                b, n, h//2, w//2, [feat_2d_2s], self.use_diff)
            feat_2d_4s = self.Feat2DPostProcess(
                b, n, h//4, w//4, [feat_2d_4s], self.use_diff)
        if self.level >= 2:
            feat_2d_8s = self.Feat2DPostProcess(
                b, n, h//8, w//8, [feat_2d_8s], self.use_diff)
        if self.level >= 3:
            feat_2d_16s = self.Feat2DPostProcess(
                b, n, h//16, w//16, [feat_2d_16s], self.use_diff)
        if self.level >= 4:
            feat_2d_32s = self.Feat2DPostProcess(
                b, n, h//32, w//32, [feat_2d_32s], self.use_diff)

        return feat_2d_2s, feat_2d_4s, feat_2d_8s, feat_2d_16s, feat_2d_32s

    def Use3DEncoder(self, x):
        b, n, c, h, w = x.shape

        feat_3d_2s, feat_3d_4s, feat_3d_8s, feat_3d_16s, feat_3d_32s = self.encoder3d(
            x.permute(0, 2, 1, 3, 4))

        return [feat_3d_2s], [feat_3d_4s], [feat_3d_8s], [feat_3d_16s], [feat_3d_32s]

    def CreateImgsDiff(self, imgs):
        # get diff_img
        b, n, c, h, w = imgs.shape
        diff_imgs = [imgs[:, 0, :, :, :]]
        for i in range(1, n):
            cur_img = imgs[:, i, :, :, :] - imgs[:, 0, :, :, :]
            diff_imgs.append(cur_img)
        diff_imgs = torch.stack(diff_imgs, dim=1)

        return diff_imgs

    def Debug_ShowImgsInfo(self, imgs, diff_imgs):
        # 用于show HSV图像
        img = imgs[0]
        diff_img = diff_imgs[0]

        logging.debug("imgs.shape={}, diff_imgs.shape={}".format(
            imgs.shape, diff_imgs.shape))
        logging.debug("img.shape={}, diff_img.shape={}".format(
            img.shape, diff_img.shape))

        img_0 = diff_img[0]  # c, h, w
        img_1 = diff_img[1]
        img_4 = diff_img[4]

        img_0 = img_0.permute(1, 2, 0)
        img_1 = img_1.permute(1, 2, 0)
        img_4 = img_4.permute(1, 2, 0)

        img_0 = img_0.cpu().detach().numpy()
        img_1 = img_1.cpu().detach().numpy()
        img_4 = img_4.cpu().detach().numpy()

        img_0 = (img_0 - img_0.min()) / (img_0.max() - img_0.min())
        img_1 = (img_1 - img_1.min()) / (img_1.max() - img_1.min())
        img_4 = (img_4 - img_4.min()) / (img_4.max() - img_4.min())

        img_0 = img_0 * 255
        img_1 = img_1 * 255
        img_4 = img_4 * 255

        img_0 = img_0.astype(np.uint8)
        img_1 = img_1.astype(np.uint8)
        img_4 = img_4.astype(np.uint8)

        img_0 = cv2.cvtColor(img_0, cv2.COLOR_RGB2HSV)
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2HSV)
        img_4 = cv2.cvtColor(img_4, cv2.COLOR_RGB2HSV)

        img_0 = np.concatenate(
            (img_0[:, :, 0], img_0[:, :, 1], img_0[:, :, 2]), axis=0)
        img_1 = np.concatenate(
            (img_1[:, :, 0], img_1[:, :, 1], img_1[:, :, 2]), axis=0)
        img_4 = np.concatenate(
            (img_4[:, :, 0], img_4[:, :, 1], img_4[:, :, 2]), axis=0)

        img = np.concatenate((img_0, img_1, img_4), axis=1)

        cv2.imshow("img", img)
        cv2.waitKey(0)

    def forward(self, imgs, focal_dists):
        # logging.debug("begin DFV forward, imgs.shape={}, focal_dists.shape={}".format(
        #     imgs.shape, focal_dists.shape))
        # diff_imgs = self.CreateImgsDiff(imgs)
        diff_imgs = imgs
        x = imgs
        b, n, c, h, w = x.shape

        # logging.debug("x.shape={}".format(x.shape))

        # self.Debug_ShowImgsInfo(imgs, diff_imgs)

        if self.encoder2d is not None:
            feat_2s, feat_4s, feat_8s, feat_16s, feat_32s = self.Use2DEncoder(
                x)
        elif self.encoder3d is not None:
            feat_2s, feat_4s, feat_8s, feat_16s, feat_32s = self.Use3DEncoder(
                x)

        upsample2d = nn.Upsample(size=(h, w), mode="bilinear")
        upsample3d = nn.Upsample(
            size=(focal_dists.shape[1], h, w), mode="trilinear")
        up_dims = [2, 2, 3, 3, 3]

        cost_l4 = self.decoder_l4(
            feat_2s[0], feat_4s[0], feat_8s[0], feat_16s[0], feat_32s[0])
        pred_l4, std_l4, cost_l4 = self.UpAndReg(
            cost_l4, focal_dists, up_dims, upsample2d, upsample3d, self.soft_max)

        pred_2s = [pred_l4[0]]
        pred_4s = [pred_l4[1]]
        pred_8s = [pred_l4[2]]
        pred_16s = [pred_l4[3]]
        pred_32s = [pred_l4[4]]

        std_2s = [std_l4[0]]
        std_4s = [std_l4[1]]
        std_8s = [std_l4[2]]
        std_16s = [std_l4[3]]
        std_32s = [std_l4[4]]

        cost_2s = [cost_l4[0]]
        cost_4s = [cost_l4[1]]
        cost_8s = [cost_l4[2]]
        cost_16s = [cost_l4[3]]
        cost_32s = [cost_l4[4]]

        return diff_imgs, [pred_2s, pred_4s, pred_8s, pred_16s, pred_32s], [std_2s, std_4s, std_8s, std_16s, std_32s], [cost_2s, cost_4s, cost_8s, cost_16s, cost_32s]
