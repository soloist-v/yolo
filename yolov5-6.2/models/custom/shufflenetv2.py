import torch
from torch import nn
from models.common import Conv


class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,W] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


# ---------------------------- ShuffleBlock start -------------------------------

# 通道重排，跨group信息交流
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class conv_bn_relu_maxpool(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out
        super(conv_bn_relu_maxpool, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(c2),
        #     nn.ReLU(inplace=True),
        # )
        self.conv = Conv(c1, c2, k=3, s=2, p=1, act=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        return self.maxpool(self.conv(x))


class Shuffle_Block(nn.Module):
    def __init__(self, inp, oup, stride):
        super(Shuffle_Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            # self.branch1 = nn.Sequential(
            #     self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
            #     nn.BatchNorm2d(inp),
            #     nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            #     nn.BatchNorm2d(branch_features),
            #     nn.ReLU(inplace=True),
            # )
            self.branch1 = nn.Sequential(
                Conv(inp, inp, k=3, s=self.stride, p=1, g=inp, act=False, bias=True),
                Conv(inp, branch_features, k=1, s=1, p=0, act=True),
            )

        # self.branch2 = nn.Sequential(
        #     nn.Conv2d(inp if (self.stride > 1) else branch_features,
        #               branch_features, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(branch_features),
        #     nn.ReLU(inplace=True),
        #     self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
        #     nn.BatchNorm2d(branch_features),
        #     nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(branch_features),
        #     nn.ReLU(inplace=True),
        # )
        self.branch2 = nn.Sequential(
            Conv(inp if (self.stride > 1) else branch_features,
                 branch_features, k=1, s=1, p=0),
            Conv(branch_features, branch_features, k=3, s=self.stride, p=1, g=branch_features),
            Conv(branch_features, branch_features, k=1, s=1, p=0),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)  # 按照维度1进行split
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

# ---------------------------- ShuffleBlock end --------------------------------
