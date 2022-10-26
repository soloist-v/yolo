import torch
from torch import nn
from models.conv import Conv


# ---------------------------- MobileBlock start -------------------------------
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        # Squeeze操作
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation操作(FC+ReLU+FC+Sigmoid)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)  # 学习到的每一channel的权重
        return x * y


class conv_bn_hswish(nn.Module):
    """
    This equals to
    def conv_3x3_bn(inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            h_swish()
        )
    """

    def __init__(self, c1, c2, stride):
        super(conv_bn_hswish, self).__init__()
        # self.conv = nn.Conv2d(c1, c2, 3, stride, 1, bias=False)
        # self.bn = nn.BatchNorm2d(c2)
        # self.act = h_swish()
        self.conv = Conv(c1, c2, 3, stride, 1, act=h_swish())

    def forward(self, x):
        # return self.act(self.bn(self.conv(x)))
        return self.conv(x)


class MobileNet_Block(nn.Module):
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs):
        super(MobileNet_Block, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        # 输入通道数=扩张通道数 则不进行通道扩张
        # if inp == hidden_dim:
        #     self.conv = nn.Sequential(
        #         # dw
        #         nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
        #                   bias=False),
        #         nn.BatchNorm2d(hidden_dim),
        #         h_swish() if use_hs else nn.ReLU(inplace=True),
        #         # Squeeze-and-Excite
        #         SELayer(hidden_dim) if use_se else nn.Sequential(),
        #         # pw-linear
        #         nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        #         nn.BatchNorm2d(oup),
        #     )
        # else:
        #     # 否则 先进行通道扩张
        #     self.conv = nn.Sequential(
        #         # pw
        #         nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
        #         nn.BatchNorm2d(hidden_dim),
        #         h_swish() if use_hs else nn.ReLU(inplace=True),
        #         # dw
        #         nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
        #                   bias=False),
        #         nn.BatchNorm2d(hidden_dim),
        #         # Squeeze-and-Excite
        #         SELayer(hidden_dim) if use_se else nn.Sequential(),
        #         h_swish() if use_hs else nn.ReLU(inplace=True),
        #         # pw-linear
        #         nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        #         nn.BatchNorm2d(oup),
        #     )
        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                Conv(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, g=hidden_dim,
                     act=h_swish() if use_hs else nn.ReLU(inplace=True)),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                # pw-linear
                Conv(hidden_dim, oup, 1, 1, 0, bias=False, act=False),
            )
        else:
            # 否则 先进行通道扩张
            self.conv = nn.Sequential(
                # pw
                Conv(inp, hidden_dim, 1, 1, 0, bias=False, act=h_swish() if use_hs else nn.ReLU(inplace=True)),
                # dw
                Conv(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, g=hidden_dim,
                     bias=False, act=False),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                Conv(hidden_dim, oup, 1, 1, 0, bias=False, act=False),
            )

    def forward(self, x):
        y = self.conv(x)
        if self.identity:
            return x + y
        else:
            return y

# ---------------------------- MobileBlock end ---------------------------------
