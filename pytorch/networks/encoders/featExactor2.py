from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import logging

from networks.basic_modules import BasicBlock, ResBlock

# code adopted from https://github.com/nianticlabs/monodepth2/blob/master/networks/resnet_encoder.py


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # , dilation=2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock,
                  50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(
        block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(
            models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class FeatExactor(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers=18, pretrained=True):
        super(FeatExactor, self).__init__()

        if pretrained:
            logging.info(
                "using pretrained resnet with num_layers={}".format(num_layers))

        num_input_images = 1

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError(
                "{} is not a valid number of resnet layers".format(num_layers))

        # 测试结果：在这里添加alter_conv1=ResBlock.Conv2d(kernel_size=3,stride=2,padding=1)，
        # 若直接用alter_conv1替换conv1并直接训练，会导致MSE大幅上升(MSE=6.28e-4@e65)
        # 猜想可能需要在替换后，在ImageNet上再训练一段时间，再迁移到这里来
        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(
                num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](weights='DEFAULT')

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        # 测试结果：UNet3Plus无SPP && num_layers=18 && featExactor2无SPP，MSE=4.76e-4@e140，对比MSE=4.45@e140
        self.spp_32 = self.CreateSPPConv(
            self.num_ch_enc[-1], self.num_ch_enc[-1])

        # self.pyramid_pooling = pyramidPooling(
        #     512, None, fusion_mode='sum', model_name='icnet')
        # Iconvs
        self.x_32_to_16 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            BasicBlock.Conv2dBnReLU(in_c=512, out_c=256, kernel_size=3, stride=1, padding=1))
        self.fuse_16 = BasicBlock.Conv2dBnReLU(
            in_c=512, out_c=256, kernel_size=3, stride=1, padding=1, norm_type="BN")

        self.x_16_to_8 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            BasicBlock.Conv2dBnReLU(in_c=256, out_c=128, kernel_size=3, stride=1, padding=1))
        self.fuse_8 = BasicBlock.Conv2dBnReLU(
            in_c=256, out_c=128, kernel_size=3, stride=1, padding=1, norm_type="BN")

        self.x_8_to_4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            BasicBlock.Conv2dBnReLU(in_c=128, out_c=64, kernel_size=3, stride=1, padding=1))
        self.fuse_4 = BasicBlock.Conv2dBnReLU(
            in_c=128, out_c=64, kernel_size=3, stride=1, padding=1, norm_type="BN")

        self.x_4_to_2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            BasicBlock.Conv2dBnReLU(in_c=64, out_c=64, kernel_size=3, stride=1, padding=1))
        self.fuse_2 = BasicBlock.Conv2dBnReLU(
            in_c=128, out_c=64, kernel_size=3, stride=1, padding=1, norm_type="BN")

        self.proj_32 = BasicBlock.Conv2dBnReLU(
            in_c=512, out_c=128, kernel_size=1, stride=1, padding=0, norm_type="BN")
        self.proj_16 = BasicBlock.Conv2dBnReLU(
            in_c=256, out_c=64, kernel_size=1, stride=1, padding=0, norm_type="BN")
        self.proj_8 = BasicBlock.Conv2dBnReLU(
            in_c=128, out_c=32, kernel_size=1, stride=1, padding=0, norm_type="BN")
        self.proj_4 = BasicBlock.Conv2dBnReLU(
            in_c=64, out_c=16, kernel_size=1, stride=1, padding=0, norm_type="BN")
        self.proj_2 = BasicBlock.Conv2dBnReLU(
            in_c=64, out_c=16, kernel_size=1, stride=1, padding=0, norm_type="BN")

    def CreateSPPConv(self, in_c, out_c):
        # TODO: IN / BN 哪个更合适？
        return torch.nn.ModuleList([
            BasicBlock.Conv2dBnReLU(
                in_c, out_c, kernel_size=1, stride=1, padding=0, norm_type="BN"),
            BasicBlock.Conv2dBnReLU(
                in_c, out_c, kernel_size=1, stride=1, padding=0, norm_type="BN"),
            BasicBlock.Conv2dBnReLU(
                in_c, out_c, kernel_size=1, stride=1, padding=0, norm_type="BN"),
            BasicBlock.Conv2dBnReLU(
                in_c, out_c, kernel_size=1, stride=1, padding=0, norm_type="BN")
        ])

    def SPP(self, x, pool_convs):
        _, _, h, w = x.shape
        res = x
        for i, pool_size in enumerate(np.linspace(1, min(h, w)//2, 4, dtype=int)):
            kernel_size = (int(h/pool_size), int(w/pool_size))
            out = F.avg_pool2d(x, kernel_size, stride=kernel_size)
            out = pool_convs[i](out)
            out = F.upsample(out, size=(h, w), mode='bilinear')
            res = res + 0.25*out
        res = F.leaky_relu(res/2., inplace=True)

        return res

    def forward(self, input_image):
        self.features = []
        x = input_image.clone()

        # H, W -> H/2, W/2
        x_2 = self.encoder.conv1(x)
        x_2 = self.encoder.bn1(x_2)
        x_2 = self.encoder.relu(x_2)

        # H/2, W/2 -> H/4, W/4
        x_4 = self.encoder.maxpool(x_2)

        # H/4, W/4 -> H/16, W/16
        x_4 = self.encoder.layer1(x_4)
        x_8 = self.encoder.layer2(x_4)
        x_16 = self.encoder.layer3(x_8)
        x_32 = self.encoder.layer4(x_16)

        # SPP会减慢收敛速度，11轮MIN_MSE=6.97e-4
        x_32 = self.SPP(x_32, self.spp_32)

        up_x_16 = self.x_32_to_16(x_32)
        x_16 = self.fuse_16(torch.cat([x_16, up_x_16], dim=1))

        up_x_8 = self.x_16_to_8(x_16)
        x_8 = self.fuse_8(torch.cat([x_8, up_x_8], dim=1))

        up_x_4 = self.x_8_to_4(x_8)
        x_4 = self.fuse_4(torch.cat([x_4, up_x_4], dim=1))

        up_x_2 = self.x_4_to_2(x_4)
        x_2 = self.fuse_2(torch.cat([x_2, up_x_2], dim=1))

        x_32 = self.proj_32(x_32)
        x_16 = self.proj_16(x_16)
        x_8 = self.proj_8(x_8)
        x_4 = self.proj_4(x_4)
        x_2 = self.proj_2(x_2)

        return x_2, x_4, x_8, x_16, x_32
