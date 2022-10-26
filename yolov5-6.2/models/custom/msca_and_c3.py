from torch import nn
from models.common import Conv, Bottleneck


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)
        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, in_, out_, n=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut and in_ == out_
        self.d_model = in_
        self.proj_1 = Conv(in_, in_, 1)
        self.m = nn.Sequential(*(Bottleneck(in_, in_, shortcut, e=1.0) for _ in range(max(n - 1, 0))))
        self.spatial_gating_unit = AttentionModule(in_)
        self.proj_2 = Conv(in_, out_, 1)

    def forward(self, x):
        shortcut = x
        x = self.proj_1(x)
        x = self.m(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.shortcut:
            x = x + shortcut
        return x
