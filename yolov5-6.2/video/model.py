import math

import torch
from torch import nn
import numpy as np
import cv2

from layers.utils import fuse_conv_and_bn_3d


class Conv3D(nn.Module):
    def __init__(self, in_, out_, k, s, p, bn=True, act=None, bias=True):
        super(Conv3D, self).__init__()
        self.conv = nn.Conv3d(in_, out_, k, s, p, bias=bias)
        self.bn = nn.BatchNorm3d(out_) if bn else nn.Identity()
        self.act = act if act else nn.Identity()
        self.is_fused = False

    def fuse(self):
        if not isinstance(self.bn, nn.Identity):
            self.conv = fuse_conv_and_bn_3d(self.conv, self.bn)
        self.is_fused = True

    def train(self, mode: bool = True):
        if (not mode) and (not self.is_fused):
            self.fuse()
        return super().train(mode)

    def forward(self, x):
        if not self.training:
            return self.act(self.conv(x))
        return self.act(self.bn(self.conv(x)))


class Residual3D(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1):
        super().__init__()
        self.conv1 = Conv3D(in_channels, in_channels, k=1, p=0, s=1, act=nn.ReLU())
        self.conv2 = Conv3D(in_channels, in_channels, k=3, p=1, s=strides, act=nn.ReLU())
        self.conv3 = Conv3D(in_channels, out_channels, k=1, p=0, s=1)
        if strides != 1 or in_channels != out_channels:
            self.shortcut = Conv3D(in_channels, out_channels, k=1, p=0, s=strides)
        else:
            self.shortcut = nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        shortcut = self.shortcut(x)
        return self.act(shortcut + y)


class Residual3DWithout1x1(nn.Module):
    def __init__(self, in_channels, div):
        super().__init__()
        stride = 1
        stage1_channel = in_channels // div
        self.conv1 = Conv3D(in_channels, stage1_channel, k=1, s=1, p=0, act=nn.ReLU())
        self.conv2 = Conv3D(stage1_channel, stage1_channel, k=3, s=stride, p=1, act=nn.ReLU())
        self.conv3 = Conv3D(stage1_channel, in_channels, k=1, s=1, p=0)
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        return self.act(x + y)


class ResNet3D(nn.Module):
    def __init__(self, in_channels=3, stem_channels=256, out_channels=(512, 768, 512, 256),
                 deep_num=(2, 2, 2, 2)):
        super().__init__()
        # src-> 1/2
        self.stem = Conv3D(in_channels, stem_channels, 3, 2, 1, act=nn.SiLU())
        self.stages = nn.ModuleList()
        last_channels = stem_channels
        for i, channels in enumerate(out_channels, 0):
            layer1 = Residual3D(last_channels, channels, 2)
            layer2 = nn.Sequential(*(Residual3DWithout1x1(channels, channels) for i in range(deep_num[0])))
            stage = nn.Sequential(layer1, layer2)
            self.stages.append(stage)
            last_channels = channels

    def forward(self, x):
        output = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            output.append(x)
        return output


class ClsHead(nn.Module):
    def __init__(self, in_features, cls_num):
        super().__init__()
        self.cls = nn.Linear(in_features, cls_num)

    def forward(self, x):
        # [1, 1024, 20, 20, 2]
        B, C, H, W, D = map(int, x.shape)
        x = x.view(B, C * H * W * D)
        print(x.shape)
        out = self.cls(x).sigmoid()
        return out


class YoloVideo(nn.Module):
    """
    可以按照时间长短分成多个模型
    模型1: 最短时间
    模型2: 中间时间
    模型3: 最长时间
    也可以直接一个模型分多个时间anchor头预测, 这就要求输入的时间更长
    先按照分模型尝试：2 4 6 8 四个模型 每秒按照16 帧计算 --> 32 64
    """

    def __init__(self, in_channels, in_h, in_w, in_d, cls_num, last_channel=1024):
        super().__init__()
        # x -> 640 640 3 32倍下采样
        # B, 512, 20, 20, 16
        for i in range(5):
            in_h = int((in_h - 3 + 2 * 1) / 2 + 1)
            in_w = int((in_w - 3 + 2 * 1) / 2 + 1)
            in_d = int((in_d - 3 + 2 * 1) / 2 + 1)
        self.backbone = ResNet3D(in_channels, 256, (256, 512, 512, last_channel), (1, 1, 1, 1))
        self.head = ClsHead(last_channel * in_h * in_w * in_d, cls_num)

    def forward(self, x):
        x = self.backbone(x)
        out = self.head(x[-1])
        return out


if __name__ == '__main__':
    import time

    # 512 512 3
    device = "cuda:0"
    x = torch.rand(1, 896, 44, 80, 64).to(device).half()
    model = YoloVideo(in_channels=896, in_h=44, in_w=80, in_d=64, cls_num=10)
    model.eval()
    model = model.to(device)
    model = model.half()
    for i in range(20):
        t0 = time.time()
        pred = model(x)
        print(time.time() - t0)
        # print(pred.shape)
