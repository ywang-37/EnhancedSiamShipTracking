import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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

class MDAM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(MDAM, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_c = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
        self.avg_pool_c = nn.AdaptiveAvgPool2d(1)

        self.linear1 = nn.Linear(in_channels, in_channels // reduction, bia=False)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(in_channels // reduction, in_channels, bia=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        
        b, c, _, _ = x.size()
        _, _, h, w = x.size()
        # GAP
        h_gap = self.pool_h(x)
        w_gap = self.pool_w(x).permute(0, 1, 3, 2)
        c_gap = self.avg_pool_c(x).view(b, c)
        y = torch.cat([h_gap, w_gap], dim=2)
        y = self.act(self.bn1(self.conv1(y))) 
        h_gap, w_gap = torch.split(y, [h, w], dim=2)
        w_gap = w_gap.permute(0, 1, 3, 2)

        # Encoder-Decoder
        h_para = self.conv_h(h_gap).sigmoid()
        w_para = self.conv_w(w_gap).sigmoid()
        c_para = self.sigmoid(self.linear2(self.relu(self.linear1(c_gap))))
        c_para = c_para.view(b, c, 1, 1)
        
        # Rescale
        out = identity * w_para * h_para + identity * c_para.expand_as(identity)

        return out