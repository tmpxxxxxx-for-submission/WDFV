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
from networks.encoders import featExactor2
from networks.attentions import SAGAN_Attention


class Unet3PlusEncoder2D(nn.Module):
    def __init__(self, level=4):
        super().__init__()

        assert level == 4
        self.level = level
        self.up_c = 32
        self.level_c = [32, 64, 128, 256, 512]

        SA_Attention = SAGAN_Attention.SAGAN_Attention

        logging.debug(
            "creating Unet3PlusEncoder2D: params={}".format(locals()))

        if self.level >= 1:
            # TODO: down_2的attention会不会有降噪的功能？
            self.down_2 = self.ResLayer2D(
                3, self.level_c[0], stride=2, block_num=2, reverse_attention=False)
            self.down_4 = self.ResLayer2D(
                self.level_c[0], self.level_c[1], stride=2, block_num=2, reverse_attention=False)
            # self.sa_attention_4 = SA_Attention(self.level_c[1])
            logging.debug("Unet3PlusEncoder2D: level 1 conv part created")

        if self.level >= 2:
            self.down_8 = self.ResLayer2D(
                self.level_c[1], self.level_c[2], stride=2, block_num=2, reverse_attention=False)
            # self.sa_attention_8 = SA_Attention(self.level_c[2])
            logging.debug("Unet3PlusEncoder2D: level 2 conv part created")

        if self.level >= 3:
            self.down_16 = self.ResLayer2D(
                self.level_c[2], self.level_c[3], stride=2, block_num=2, reverse_attention=False)
            # self.sa_attention_16 = SA_Attention(self.level_c[3])
            logging.debug("Unet3PlusEncoder2D: level 3 conv part created")

        if self.level >= 4:
            self.down_32 = self.ResLayer2D(
                self.level_c[3], self.level_c[4], stride=2, block_num=2, reverse_attention=False)
            # self.sa_attention_32 = SA_Attention(self.level_c[4])
            logging.debug("Unet3PlusEncoder2D: level 4 conv part created")

        if self.level >= 4:
            self.rescale_32_to_16 = self.UpsampleBlk(
                self.level_c[4], self.up_c, scale_factor=2)
            self.rescale_32_to_8 = self.UpsampleBlk(
                self.level_c[4], self.up_c, scale_factor=4)
            self.rescale_32_to_4 = self.UpsampleBlk(
                self.level_c[4], self.up_c, scale_factor=8)
            self.rescale_32_to_2 = self.UpsampleBlk(
                self.level_c[4], self.up_c, scale_factor=16)

            logging.debug("Unet3PlusEncoder2D: level 4 rescale part created")

        if self.level >= 3:
            self.rescale_16_to_16 = self.FusionBlk(self.level_c[3], self.up_c)
            self.rescale_16_to_8 = self.UpsampleBlk(
                self.level_c[3], self.up_c, scale_factor=2)
            self.rescale_16_to_4 = self.UpsampleBlk(
                self.level_c[3], self.up_c, scale_factor=4)
            self.rescale_16_to_2 = self.UpsampleBlk(
                self.level_c[3], self.up_c, scale_factor=8)
            logging.debug("Unet3PlusEncoder2D: level 3 rescale part created")

        if self.level >= 2:
            self.rescale_8_to_16 = self.DownsampleBlk(
                self.level_c[2], self.up_c, scale_factor=2)
            self.rescale_8_to_8 = self.FusionBlk(self.level_c[2], self.up_c)
            self.rescale_8_to_4 = self.UpsampleBlk(
                self.level_c[2], self.up_c, scale_factor=2)
            self.rescale_8_to_2 = self.UpsampleBlk(
                self.level_c[2], self.up_c, scale_factor=4)
            logging.debug("Unet3PlusEncoder2D: level 2 rescale part created")

        if self.level >= 1:
            self.rescale_4_to_16 = self.DownsampleBlk(
                self.level_c[1], self.up_c, scale_factor=4)
            self.rescale_4_to_8 = self.DownsampleBlk(
                self.level_c[1], self.up_c, scale_factor=2)
            self.rescale_4_to_4 = self.FusionBlk(self.level_c[1], self.up_c)
            self.rescale_4_to_2 = self.UpsampleBlk(
                self.level_c[1], self.up_c, scale_factor=2)

            self.rescale_2_to_16 = self.DownsampleBlk(
                self.level_c[0], self.up_c, scale_factor=8)
            self.rescale_2_to_8 = self.DownsampleBlk(
                self.level_c[0], self.up_c, scale_factor=4)
            self.rescale_2_to_4 = self.DownsampleBlk(
                self.level_c[0], self.up_c, scale_factor=2)
            self.rescale_2_to_2 = self.FusionBlk(self.level_c[0], self.up_c)
            logging.debug("Unet3PlusEncoder2D: level 1 rescale part created")

        self.res_conv_16 = ResBlock.ResBasicBlock2D(self.up_c*5, self.up_c*5)
        self.res_conv_8 = ResBlock.ResBasicBlock2D(self.up_c*5, self.up_c*5)
        self.res_conv_4 = ResBlock.ResBasicBlock2D(self.up_c*5, self.up_c*5)
        self.res_conv_2 = ResBlock.ResBasicBlock2D(self.up_c*5, self.up_c*5)

        # logging.debug("proj part created")

        logging.debug("Unet3PlusEncoder2D: finish creating Unet3PlusEncoder2D")

    def ResLayer2D(self, in_c, out_c, kernel_size=3, stride=1, padding=1, block_num=2, reverse_attention=False):
        blks = [ResBlock.ResBasicBlock2D(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding,
                                         reverse_attention=None)]
        for cnt in range(1, block_num - 1):
            blks += [ResBlock.ResBasicBlock2D(out_c, out_c, stride=1,
                                              reverse_attention=None)]
        blks += [ResBlock.ResBasicBlock2D(out_c, out_c, stride=1,
                                          reverse_attention=reverse_attention)]
        return nn.Sequential(*blks)

    def DownsampleBlk(self, in_c, out_c, scale_factor=2, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=scale_factor*2-1,
                         stride=scale_factor, padding=scale_factor//2),
            self.FusionBlk(in_c, out_c, kernel_size=kernel_size,
                           stride=stride, padding=padding),
        )

    def UpsampleBlk(self, in_c, out_c, scale_factor=2, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="bilinear"),
            self.FusionBlk(in_c, out_c, kernel_size=kernel_size,
                           stride=stride, padding=padding),
        )

    def FusionBlk(self, in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            BasicBlock.Conv2dBnReLU(
                in_c, out_c, kernel_size, stride, padding, bias),
            BasicBlock.Conv2dBnReLU(
                out_c, out_c, kernel_size, stride, padding, bias),
        )

    def forward(self, x):
        # b*n,3,h,w

        if self.level >= 1:
            x_2 = self.down_2(x)  # c = 64
            x_4 = self.down_4(x_2)  # c = 128
            # x_4, attention_map_4 = self.sa_attention_4(x_4)

        if self.level >= 2:
            x_8 = self.down_8(x_4)  # c = 256
            # x_8, attention_map_8 = self.sa_attention_8(x_8)

        if self.level >= 3:
            x_16 = self.down_16(x_8)  # c = 512
            # x_16, attention_map_16 = self.sa_attention_16(x_16)

        if self.level >= 4:
            x_32 = self.down_32(x_16)  # c = 512
            # x_32, attention_map_32 = self.sa_attention_32(x_32)

        if self.level >= 4:
            x_32_to_16 = self.rescale_32_to_16(x_32)
            x_32_to_8 = self.rescale_32_to_8(x_32)
            x_32_to_4 = self.rescale_32_to_4(x_32)
            x_32_to_2 = self.rescale_32_to_2(x_32)

        if self.level >= 3:
            x_16_to_16 = self.rescale_16_to_16(x_16)
            x_16_to_8 = self.rescale_16_to_8(x_16)
            x_16_to_4 = self.rescale_16_to_4(x_16)
            x_16_to_2 = self.rescale_16_to_2(x_16)

        if self.level >= 2:
            x_8_to_16 = self.rescale_8_to_16(x_8)
            x_8_to_8 = self.rescale_8_to_8(x_8)
            x_8_to_4 = self.rescale_8_to_4(x_8)
            x_8_to_2 = self.rescale_8_to_2(x_8)

        if self.level >= 1:
            x_4_to_16 = self.rescale_4_to_16(x_4)
            x_4_to_8 = self.rescale_4_to_8(x_4)
            x_4_to_4 = self.rescale_4_to_4(x_4)
            x_4_to_2 = self.rescale_4_to_2(x_4)

            x_2_to_16 = self.rescale_2_to_16(x_2)
            x_2_to_8 = self.rescale_2_to_8(x_2)
            x_2_to_4 = self.rescale_2_to_4(x_2)
            x_2_to_2 = self.rescale_2_to_2(x_2)

        res_2 = torch.cat([x_2_to_2, x_4_to_2, x_8_to_2,
                           x_16_to_2, x_32_to_2], dim=1)
        res_4 = torch.cat([x_2_to_4, x_4_to_4, x_8_to_4,
                           x_16_to_4, x_32_to_4], dim=1)
        res_8 = torch.cat([x_2_to_8, x_4_to_8, x_8_to_8,
                           x_16_to_8, x_32_to_8], dim=1)
        res_16 = torch.cat([x_2_to_16, x_4_to_16, x_8_to_16,
                            x_16_to_16, x_32_to_16], dim=1)

        res_2 = self.res_conv_2(res_2)
        res_4 = self.res_conv_4(res_4)
        res_8 = self.res_conv_8(res_8)
        res_16 = self.res_conv_16(res_16)
        res_32 = x_32

        return res_2, res_4, res_8, res_16, res_32


################################# 3D version ###########################


class Unet3PlusEncoder3D(nn.Module):
    def __init__(self, level=4):
        super().__init__()

        self.level = level
        # 测试发现8通道不够
        # 测试结果：up_c的64与32效果差不多
        # 测试结果：[8,8,16,32,64]与[16,16,32,64,128]效果差不多
        # 测试结果：up_c=[8,8,16,32,64]效果变差了，MSE=6.51e-4@e26，对比MSE=5.49e-4@e26
        self.up_c = [16, 16, 16, 16, 16]
        # 使用res-50时，in_c[1:]*=4效果基本无
        self.in_c = [16, 16, 32, 64, 128]
        # [64,64,128,256,512]与[32,32,64,128,256]效果差不多
        self.level_c = [16, 16, 32, 64, 128]

        logging.debug(
            "creating Unet3PlusEncoder3D: params={}".format(locals()))
        logging.warning("self.level is not using!!!")

        # 使用res-18预训练模型即可，使用fcn(res-50)或res-50或fcn&&res-50都不会有更好的结果
        self.feature_extraction = featExactor2.FeatExactor()

        # 实验结果：仅enc时，block_num=2, up_c=32，最终MSE=5.73e-4
        # block_num=1即可，无需2。ResBlock自带2层conv(in_c->out_c && out_c->out_c)
        # 用上CBAM似乎收敛更快一点点
        # 实验结果：将enc_xx的block_num调整为[3,4,6,3]，MSE=7.92e-4@e12，对比MSE=6.575e-4@e12
        # 实验结果：将enc_xx的block_num调整为[3,3,3,3]，MSE=4.70e-4@e89，对比MSE=4.67e-4@e89
        # 实验结果：将fuse_xx的block_num调整为[3,4,6,3]，MSE=7.09e-4@e18，对比MSE=5.38e-4@e18
        # 实验结果：删除UNet3P的SPP后，将fuse_xx的block_num调整为[2,2,2,2]，MSE=4.61e-4@e144
        self.enc_2 = nn.Sequential(
            ResBlock.ResBasicBlock3D(
                self.in_c[0], self.level_c[0], stride=1, reverse_attention=None),
            ResBlock.ResBasicBlock3D(
                self.level_c[0], self.level_c[0], stride=1, reverse_attention=None),
        )

        self.enc_4 = nn.Sequential(
            ResBlock.ResBasicBlock3D(
                self.in_c[1], self.level_c[1], stride=1, reverse_attention=None),
            ResBlock.ResBasicBlock3D(
                self.level_c[1], self.level_c[1], stride=1, reverse_attention=None),
        )
        # self.down_4 = nn.Sequential(
        #     ResBlock.ResBasicBlock3D(
        #         self.level_c[0], self.level_c[1], stride=(1, 2, 2), reverse_attention=None),
        #     ResBlock.ResBasicBlock3D(
        #         self.level_c[1], self.level_c[1], stride=(1, 1, 1), reverse_attention=None),
        # )
        # self.fuse_4 = nn.Sequential(
        #     ResBlock.ResBasicBlock3D(
        #         self.level_c[1]*2, self.level_c[1], stride=1, reverse_attention=None),
        # )
        logging.debug("Unet3PlusEncoder3D: level 1 conv part created")

        self.enc_8 = nn.Sequential(
            ResBlock.ResBasicBlock3D(
                self.in_c[2], self.level_c[2], stride=1, reverse_attention=None),
            ResBlock.ResBasicBlock3D(
                self.level_c[2], self.level_c[2], stride=1, reverse_attention=None),
        )
        # self.down_8 = nn.Sequential(
        #     ResBlock.ResBasicBlock3D(
        #         self.level_c[1], self.level_c[2], stride=(1, 2, 2), reverse_attention=None),
        #     ResBlock.ResBasicBlock3D(
        #         self.level_c[2], self.level_c[2], stride=(1, 1, 1), reverse_attention=None),
        # )
        # self.fuse_8 = nn.Sequential(
        #     ResBlock.ResBasicBlock3D(
        #         self.level_c[2]*2, self.level_c[2], stride=1, reverse_attention=None),
        # )
        logging.debug("Unet3PlusEncoder3D: level 2 conv part created")

        self.enc_16 = nn.Sequential(
            ResBlock.ResBasicBlock3D(
                self.in_c[3], self.level_c[3], stride=1, reverse_attention=None),
            ResBlock.ResBasicBlock3D(
                self.level_c[3], self.level_c[3], stride=1, reverse_attention=None),
        )
        # self.down_16 = nn.Sequential(
        #     ResBlock.ResBasicBlock3D(
        #         self.level_c[2], self.level_c[3], stride=(1, 2, 2), reverse_attention=None),
        #     ResBlock.ResBasicBlock3D(
        #         self.level_c[3], self.level_c[3], stride=(1, 1, 1), reverse_attention=None),
        # )
        # self.fuse_16 = nn.Sequential(
        #     ResBlock.ResBasicBlock3D(
        #         self.level_c[3]*2, self.level_c[3], stride=1, reverse_attention=None),
        # )
        logging.debug("Unet3PlusEncoder3D: level 3 conv part created")

        self.enc_32 = nn.Sequential(
            ResBlock.ResBasicBlock3D(
                self.in_c[4], self.level_c[4], stride=1, reverse_attention=None),
            ResBlock.ResBasicBlock3D(
                self.level_c[4], self.level_c[4], stride=1, reverse_attention=None),
        )
        # self.down_32 = nn.Sequential(
        #     ResBlock.ResBasicBlock3D(
        #         self.level_c[3], self.level_c[4], stride=(1, 2, 2), reverse_attention=None),
        #     ResBlock.ResBasicBlock3D(
        #         self.level_c[4], self.level_c[4], stride=(1, 1, 1), reverse_attention=None),
        # )
        # self.fuse_32 = nn.Sequential(
        #     ResBlock.ResBasicBlock3D(
        #         self.level_c[4]*2, self.level_c[4], stride=1, reverse_attention=None),
        # )
        logging.debug("Unet3PlusEncoder3D: level 4 conv part created")

        # self.spp_32 = self.CreateSPPConv(self.level_c[4])

        # 修正UNet3+的结构后，空间池化金字塔能够反而会增大MSE
        # 实验结果：删除UNet3P的SPP && 损失函数全程用SmoothL1，MSE_1=4.45e-4@e213, MSE_2=4.55e-4@e178
        up_c_sum = sum(self.up_c)
        self.res_conv_32 = self.ResLayer3D(
            self.level_c[4], self.level_c[4], stride=(1, 1, 1), block_num=2, reverse_attention=None)

        self.rescale_32_to_16 = self.UpsampleBlk(
            self.level_c[4], self.up_c[4], scale_factor=2)
        self.rescale_16_to_16 = self.FusionBlk(self.level_c[3], self.up_c[3])
        self.rescale_8_to_16 = self.DownsampleBlk(
            self.level_c[2], self.up_c[2], scale_factor=2)
        self.rescale_4_to_16 = self.DownsampleBlk(
            self.level_c[1], self.up_c[1], scale_factor=4)
        self.rescale_2_to_16 = self.DownsampleBlk(
            self.level_c[0], self.up_c[0], scale_factor=8)
        self.res_conv_16 = self.ResLayer3D(
            up_c_sum, up_c_sum, stride=(1, 1, 1), block_num=2, reverse_attention=None)
        logging.debug("Unet3PlusEncoder3D: level 3 rescale part created")

        self.rescale_32_to_8 = self.UpsampleBlk(
            self.level_c[4], self.up_c[4], scale_factor=4)
        self.rescale_16_to_8 = self.UpsampleBlk(
            up_c_sum, self.up_c[3], scale_factor=2)
        self.rescale_8_to_8 = self.FusionBlk(self.level_c[2], self.up_c[2])
        self.rescale_4_to_8 = self.DownsampleBlk(
            self.level_c[1], self.up_c[1], scale_factor=2)
        self.rescale_2_to_8 = self.DownsampleBlk(
            self.level_c[0], self.up_c[0], scale_factor=4)
        self.res_conv_8 = self.ResLayer3D(
            up_c_sum, up_c_sum, stride=(1, 1, 1), block_num=2, reverse_attention=None)
        logging.debug("Unet3PlusEncoder3D: level 2 rescale part created")

        self.rescale_32_to_4 = self.UpsampleBlk(
            self.level_c[4], self.up_c[4], scale_factor=8)
        self.rescale_16_to_4 = self.UpsampleBlk(
            up_c_sum, self.up_c[3], scale_factor=4)
        self.rescale_8_to_4 = self.UpsampleBlk(
            up_c_sum, self.up_c[2], scale_factor=2)
        self.rescale_4_to_4 = self.FusionBlk(self.level_c[1], self.up_c[1])
        self.rescale_2_to_4 = self.DownsampleBlk(
            self.level_c[0], self.up_c[0], scale_factor=2)
        self.res_conv_4 = self.ResLayer3D(
            up_c_sum, up_c_sum, stride=(1, 1, 1), block_num=2, reverse_attention=None)

        self.rescale_32_to_2 = self.UpsampleBlk(
            self.level_c[4], self.up_c[4], scale_factor=16)
        self.rescale_16_to_2 = self.UpsampleBlk(
            up_c_sum, self.up_c[3], scale_factor=8)
        self.rescale_8_to_2 = self.UpsampleBlk(
            up_c_sum, self.up_c[2], scale_factor=4)
        self.rescale_4_to_2 = self.UpsampleBlk(
            up_c_sum, self.up_c[1], scale_factor=2)
        self.rescale_2_to_2 = self.FusionBlk(self.level_c[0], self.up_c[0])
        self.res_conv_2 = self.ResLayer3D(
            up_c_sum, up_c_sum, stride=(1, 1, 1), block_num=2, reverse_attention=None)
        logging.debug("Unet3PlusEncoder3D: level 1 rescale part created")

        logging.debug("Unet3PlusEncoder3D: finish creating Unet3PlusEncoder3D")

    def ResLayer3D(self, in_c, out_c, kernel_size=3, stride=1, padding=1, block_num=2, reverse_attention=False):
        blks = [ResBlock.ResBasicBlock3D(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding,
                                         reverse_attention=reverse_attention)]
        for cnt in range(1, block_num - 1):
            blks += [ResBlock.ResBasicBlock3D(out_c, out_c, stride=1,
                                              reverse_attention=None)]
        if block_num > 1:
            blks += [ResBlock.ResBasicBlock3D(out_c, out_c, stride=1,
                                              reverse_attention=reverse_attention)]
        return nn.Sequential(*blks)

    def DownsampleBlk(self, in_c, out_c, scale_factor=2, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, scale_factor*2-1, scale_factor*2-1),
                         stride=(1, scale_factor, scale_factor),
                         padding=(0, scale_factor//2, scale_factor//2)),
            self.FusionBlk(in_c, out_c, kernel_size=kernel_size,
                           stride=stride, padding=padding),
        )

    def UpsampleBlk(self, in_c, out_c, scale_factor=2, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            nn.Upsample(scale_factor=(1, scale_factor,
                                      scale_factor), mode="trilinear"),
            self.FusionBlk(in_c, out_c, kernel_size=kernel_size,
                           stride=stride, padding=padding),
        )

    def FusionBlk(self, in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            BasicBlock.Conv3dBnReLU(
                in_c, out_c, kernel_size, stride, padding, bias),
            BasicBlock.Conv3dBnReLU(
                out_c, out_c, kernel_size, stride, padding, bias),
        )

    def CreateSPPConv(self, in_c):
        # TODO: IN / BN 哪个更合适？
        out_c = in_c//4
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
        # 实验结果：令pool_d=[5,5,2,2]会略微降低性能，MSE=4.77e-4@e167，对比MSE=4.41e-4@e167
        # 实验结果：令pool_sizes = [(1,1,1), (1,2,2),(1,3,3),(2,6,6)], MSE=4.92e-4@e78，对比MSE=4.74e-4@e78
        pool_sizes = [(d, 1, 1), (d, 2, 2), (d, 3, 3), (d, 5, 5)]
        for i, pool_size in enumerate(pool_sizes):
            avg_pool3d = nn.AdaptiveAvgPool3d(pool_size)
            out = avg_pool3d(x)
            out = pool_convs[i](out)
            out = nn.functional.interpolate(out, size=(
                d, h, w), mode='trilinear', align_corners=True)
            res = torch.cat([res, out], dim=1)
        return res

    def CreateFeatureDiff(self, imgs):
        # get diff_img
        b, d, c, h, w = imgs.shape
        diff_imgs = [imgs[:, 0, :, :, :]]
        for i in range(1, d):
            cur_img = imgs[:, i, :, :, :] - imgs[:, i-1, :, :, :]
            diff_imgs.append(cur_img)
        diff_imgs = torch.stack(diff_imgs, dim=1)

        return diff_imgs

    def CreateImgsDiff(self, imgs):
        # get diff_img
        b, d, c, h, w = imgs.shape
        diff_imgs = [imgs[:, 0, :, :, :]]
        for i in range(1, d):
            cur_img = imgs[:, i, :, :, :] - imgs[:, 0, :, :, :]
            diff_imgs.append(cur_img)
        diff_imgs = torch.stack(diff_imgs, dim=1)

        return diff_imgs

    def ShowFeatureImgs(self, x_00, title="title", show_all=False):
        x_00 = x_00.clone().detach().cpu().numpy()
        feat_num = x_00.shape[0]
        row = math.ceil(math.sqrt(feat_num))

        f, axarr = plt.subplots(row, row)
        f.tight_layout()
        f.canvas.set_window_title(title)
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        show_cnt = 0
        for r in range(row):
            for c in range(row):
                axarr[r][c].imshow(x_00[show_cnt], cmap=plt.cm.jet)
                show_cnt += 1
                if show_cnt >= feat_num:
                    break
            if show_cnt >= feat_num:
                break

        if show_all:
            plt.show()
            # plt.pause(0.1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        images_stack = x.permute(0, 2, 1, 3, 4).view(b*d, c, h, w)
        x_2, x_4, x_8, x_16, x_32 = self.feature_extraction(images_stack)

        x_2 = x_2.view(b, d, -1, h//2, w//2)
        x_4 = x_4.view(b, d, -1, h//4, w//4)
        x_8 = x_8.view(b, d, -1, h//8, w//8)
        x_16 = x_16.view(b, d, -1, h//16, w//16)
        x_32 = x_32.view(b, d, -1, h//32, w//32)

        # self.ShowFeatureImgs(x_4[0, 0, :, :, :], "2d feat")

        x_2 = self.CreateFeatureDiff(x_2).permute(0, 2, 1, 3, 4)
        x_4 = self.CreateFeatureDiff(x_4).permute(0, 2, 1, 3, 4)
        x_8 = self.CreateFeatureDiff(x_8).permute(0, 2, 1, 3, 4)
        x_16 = self.CreateFeatureDiff(x_16).permute(0, 2, 1, 3, 4)
        x_32 = self.CreateFeatureDiff(x_32).permute(0, 2, 1, 3, 4)

        # 实验结果：非常不建议引入x_diff，会导致MSE降不到5e-4以下
        # x_diff = self.CreateImgsDiff(
        #     x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)

        x_2 = self.enc_2(x_2)  # c = 64
        # down_2 = self.down_2(x_diff)
        # down_2 = x_2
        # x_2 = self.fuse_2(torch.cat([x_2, down_2], dim=1))

        x_4 = self.enc_4(x_4)  # c = 128
        # down_4 = self.down_4(x_2)
        # x_4 = self.fuse_4(torch.cat([x_4, down_4], dim=1))

        x_8 = self.enc_8(x_8)  # c = 256
        # down_8 = self.down_8(x_4)
        # x_8 = self.fuse_8(torch.cat([x_8, down_8], dim=1))

        x_16 = self.enc_16(x_16)  # c = 512
        # down_16 = self.down_16(x_8)
        # x_16 = self.fuse_16(torch.cat([x_16, down_16], dim=1))

        x_32 = self.enc_32(x_32)  # c = 512
        # down_32 = self.down_32(x_16)
        # x_32 = self.fuse_32(torch.cat([x_32, down_32], dim=1))

        # x_2 = x_2 + down_2
        # x_4 = x_4 + down_4
        # x_8 = x_8 + down_8
        # x_16 = x_16 + down_16
        # x_32 = x_32 + down_32

        # res_32 = self.SPP(x_32, self.spp_32)
        res_32 = self.res_conv_32(x_32)

        x_32_to_16 = self.rescale_32_to_16(res_32)
        x_16_to_16 = self.rescale_16_to_16(x_16)
        x_8_to_16 = self.rescale_8_to_16(x_8)
        x_4_to_16 = self.rescale_4_to_16(x_4)
        x_2_to_16 = self.rescale_2_to_16(x_2)
        res_16 = torch.cat([x_2_to_16, x_4_to_16, x_8_to_16,
                            x_16_to_16, x_32_to_16], dim=1)
        res_16 = self.res_conv_16(res_16)

        x_32_to_8 = self.rescale_32_to_8(res_32)
        x_16_to_8 = self.rescale_16_to_8(res_16)
        x_8_to_8 = self.rescale_8_to_8(x_8)
        x_4_to_8 = self.rescale_4_to_8(x_4)
        x_2_to_8 = self.rescale_2_to_8(x_2)
        res_8 = torch.cat([x_2_to_8, x_4_to_8, x_8_to_8,
                           x_16_to_8, x_32_to_8], dim=1)
        res_8 = self.res_conv_8(res_8)

        x_32_to_4 = self.rescale_32_to_4(res_32)
        x_16_to_4 = self.rescale_16_to_4(res_16)
        x_8_to_4 = self.rescale_8_to_4(res_8)
        x_4_to_4 = self.rescale_4_to_4(x_4)
        x_2_to_4 = self.rescale_2_to_4(x_2)
        res_4 = torch.cat([x_2_to_4, x_4_to_4, x_8_to_4,
                           x_16_to_4, x_32_to_4], dim=1)
        res_4 = self.res_conv_4(res_4)

        x_32_to_2 = self.rescale_32_to_2(res_32)
        x_16_to_2 = self.rescale_16_to_2(res_16)
        x_8_to_2 = self.rescale_8_to_2(res_8)
        x_4_to_2 = self.rescale_4_to_2(res_4)
        x_2_to_2 = self.rescale_2_to_2(x_2)
        res_2 = torch.cat([x_2_to_2, x_4_to_2, x_8_to_2,
                           x_16_to_2, x_32_to_2], dim=1)
        res_2 = self.res_conv_2(res_2)

        # self.ShowFeatureImgs(res_2[0, :, 0, :, :], "res 0 spp")
        # self.ShowFeatureImgs(res_2[0, :, 1, :, :], "res 1 spp")
        # self.ShowFeatureImgs(res_2[0, :, 2, :, :], "res 2 spp")
        # self.ShowFeatureImgs(res_2[0, :, 3, :, :], "res 3 spp")
        # self.ShowFeatureImgs(res_2[0, :, 4, :, :], "res 4 spp", show_all=True)

        return res_2, res_4, res_8, res_16, res_32
