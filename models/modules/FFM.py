import torch
import torch.nn.functional as F
import torch.nn as nn
import math


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=False):
        super(DSConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x







class SemanticFusion(nn.Module):
    def __init__(self, high_channels, low_channels):
        super(SemanticFusion, self).__init__()
        self.conv_low = nn.Conv2d(high_channels, low_channels, 1, 1)
        self.ffmconv = nn.Sequential(
            DSConv(low_channels*2, low_channels, 3, 1),
            nn.BatchNorm2d(low_channels),
            nn.ReLU()
        )


    def forward(self, x_high, x_low):
        x_high = self.conv_low(x_high)

        x_high = torch.cat((x_low, F.interpolate(x_high, scale_factor=2, mode='bilinear', align_corners=False)), 1)

        x_high =self.ffmconv(x_high)




        return x_high


class FFM(nn.Module):
    def __init__(self, in_channels=[32, 64, 160, 256], num_classes=4):
        super(FFM, self).__init__()
        self.sf1 = SemanticFusion(in_channels[3], in_channels[2])
        self.sf2 = SemanticFusion(in_channels[2], in_channels[1])
        self.sf3 = SemanticFusion(in_channels[1], in_channels[0])

    def forward(self, x):
        x1, x2, x3, x4 = x
        outs=[]

        sf1_out = self.sf1(x4, x3)

        sf2_out = self.sf2(sf1_out, x2)

        sf3_out = self.sf3(sf2_out, x1)
        outs.append(sf3_out)
        outs.append(sf2_out)
        outs.append(sf1_out)
        outs.append(x4)


        return outs



if __name__ == '__main__':
    # 随机生成四个张量
    tensor1 = torch.randn(8, 32, 64, 64)
    tensor2 = torch.randn(8, 64, 32, 32)
    tensor3 = torch.randn(8, 160, 16, 16)
    tensor4 = torch.randn(8, 256, 8, 8)

    # 创建一个包含这四个张量的列表
    tensor_list = [tensor1, tensor2, tensor3, tensor4]


    model= msf()

    #model = ConvBNReLU(3, 64)
    #model = ConvCA(64,256)


    print(model(tensor_list).shape)

