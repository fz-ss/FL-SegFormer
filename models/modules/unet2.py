from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x1)
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)

        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # from .nam_efficientnetv2 import efficientnetv2_s
        # self.backbone = efficientnetv2_s(num_classes=1000).to(device)
        # 'b0': [32, 64, 160, 256]
        factor = 2 if bilinear else 1

        self.up1 = Up(256, 160, bilinear).to(device)
        self.up2 = Up(320, 64, bilinear).to(device)
        self.up3 = Up(128, 32, bilinear).to(device)

        self.conv = DoubleConv(64, 64, 64 // 2).to(device)


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        x0, x1, x2, x3 = x
        # 16,32,48,96,112,192
        # 192
         # 16
          # 32
        # 48
        # 112
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x0)

        x = self.conv(x)

        return x
if __name__ == '__main__':
    # block = UNet(in_channels=3, num_classes=4, base_c=32)
    # input = torch.rand(8, 3, 256, 256).cuda()
    # output = block(input)
    # print(input.size(), output.size())
    tensor1 = torch.randn(8, 32, 64, 64).cuda()
    tensor2 = torch.randn(8, 64, 32, 32).cuda()
    tensor3 = torch.randn(8, 160, 16, 16).cuda()
    tensor4 = torch.randn(8, 256, 8, 8).cuda()

    # 创建一个包含这四个张量的列表
    tensor_list = [tensor1, tensor2, tensor3, tensor4]

    model = UNet(in_channels=3, num_classes=4, base_c=32).cuda()

    # model = ConvBNReLU(3, 64)
    # model = ConvCA(64,256)

    print(model(tensor_list).shape)


