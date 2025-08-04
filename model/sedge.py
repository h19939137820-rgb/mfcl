import torch
from torch import nn
import numpy as np

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ERB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ERB, self).__init__()
        self.se_attention = SELayer(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, relu=True):
        x = self.se_attention(x)
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        if relu:
            return self.relu(x + res)
        else:
            return x + res

def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input

class SEdge(nn.Module):
    def __init__(self, dim=[128, 256, 512, 1024]):
        super(SEdge, self).__init__()
        dim0, dim1, dim2, dim3 = dim
        self.erb_db_1 = ERB(dim0, 2)
        self.erb_db_2 = ERB(dim1, 2)
        self.erb_db_3 = ERB(dim2, 2)
        self.erb_db_4 = ERB(dim3, 2)
        self.sobel_x1, self.sobel_y1 = get_sobel(dim0, 1)
        self.sobel_x2, self.sobel_y2 = get_sobel(dim1, 1)
        self.sobel_x3, self.sobel_y3 = get_sobel(dim2, 1)
        self.sobel_x4, self.sobel_y4 = get_sobel(dim3, 1)

        self.erb_trans_1 = ERB(2, 2)
        self.erb_trans_2 = ERB(2, 2)
        self.erb_trans_3 = ERB(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

    def forward(self, x):
        x0, x1, x2, x3 = x
        res1 = self.erb_db_1(run_sobel(self.sobel_x1, self.sobel_y1, x0))
        res1 = self.erb_trans_1(res1 + self.upsample(self.erb_db_2(run_sobel(self.sobel_x2, self.sobel_y2, x1))))
        res1 = self.erb_trans_2(res1 + self.upsample_4(self.erb_db_3(run_sobel(self.sobel_x3, self.sobel_y3, x2))))
        res1 = self.erb_trans_3(res1 + self.upsample_8(self.erb_db_4(run_sobel(self.sobel_x4, self.sobel_y4, x3))),
                                relu=False)
        return res1

