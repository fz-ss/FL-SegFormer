import torch
import torch.nn.functional as F
import torch.nn as nn
from .attention import LKA
class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DSConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class FPN(nn.Module):
    def     __init__(self,in_channels=[32, 64, 160, 256]):
        super(FPN,self).__init__()
        # self.ema1 = EMA(256)
        # self.ema2 = EMA(256)
        # self.ema3 = EMA(256)
        # self.ema4 = EMA(256)

        self.cbam1 = LKA(256)
        self.cbam2 = LKA(256)
        self.cbam3 = LKA(256)
        self.cbam4 = LKA(256)


        self.conv1 = nn.Conv2d(in_channels[3], in_channels[3], kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels[2], in_channels[3], kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels[1], in_channels[3], kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels[0], in_channels[3], kernel_size=1)
        #
        # self.ema1 = EMA(32)
        # self.ema1 = EMA(32)
        # self.ema2 = EMA(64)
        # self.ema3 = EMA(160)
        # self.ema4 = EMA(256)

    def forward(self,x):
        c1,c2,c3,c4 = x
        # c1(32,64,64),c2(64,32,32),c3(160,16,16),c4(256,8,8)
        p4 = self.conv1(c4)  # 得到一张特征图
        p3 = self.conv2(c3)   # 这里将conv3(c3)得第二张特征图与第一张特征图进行融合，由于尺寸不同要进行上采样
        p2 = self.conv3(c2) + F.interpolate(p3, scale_factor=2, mode='bilinear')# 同理得到三张特征图的融合图像
        p1 = self.conv4(c1) + F.interpolate(p2, scale_factor=2, mode='bilinear')
        # p3 = self.conv2(c3) + F.interpolate(p4, scale_factor=2, mode='bilinear')
        # p2 = self.conv3(c2) + F.interpolate(p3, scale_factor=2, mode='bilinear')  # 同理得到三张特征图的融合图像
        # p1 = self.conv4(c1) + F.interpolate(p2, scale_factor=2, mode='bilinear')

        p4 = self.cbam4(p4)
        p3 = self.cbam3(p3)
        p2 = self.cbam2(p2)
        p1 = self.cbam1(p1)

        # p4 = self.ema4(p4)
        # p3 = self.ema3(p3)
        # p2 = self.ema2(p2)
        # p1 = self.ema1(p1)
        # p4 = self.ema4(c4)
        # p3 = self.ema3(c3)
        # p2 = self.ema2(c2)
        # p1 = self.ema1(c1)

        return p1, p2, p3, p4



if __name__ == '__main__':
    # 随机生成四个张量
    tensor1 = torch.randn(8, 32, 64, 64)
    tensor2 = torch.randn(8, 64, 32, 32)
    tensor3 = torch.randn(8, 160, 16, 16)
    tensor4 = torch.randn(8, 256, 8, 8)

    # 创建一个包含这四个张量的列表
    tensor_list = [tensor1, tensor2, tensor3, tensor4]
    model = FPN()
    out=model(tensor_list)

