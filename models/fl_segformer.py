import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.fpn import FPN
from .modules.unet2 import UNet
from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from .modules.attention import gnconv

class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class gnconvModel(nn.Module):
    def __init__(self, inchannel=256):
        super(gnconvModel, self).__init__()

        self.conv1 = ConvModule(
            c1=256,
            c2=64,
            k=1,
        )
        self.conv2 = ConvModule(
            c1=64,
            c2=256,
            k=1,
        )

        self.part = gnconv(64)
    def forward(self, x):
        x = self.conv1(x)
        x = self.part(x)
        x = self.conv2(x)
        return  x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.fpn = FPN()
        # self.unet =UNet(in_channels=3, num_classes=4, base_c=32)
        # self.linearc4 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        # self.linearc3 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        # self.linearc2 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)
        # self.linearc1 = MLP(input_dim=embedding_dim, embed_dim=embedding_dim)


        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        self.gnconvM = gnconvModel(256)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        # out = self.unet(inputs)
        c1, c2, c3, c4 = self.fpn(inputs)
        # _c4 = self.linearc4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        # _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # _c3 = self.linearc3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        # _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # _c2 = self.linearc2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        # _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # _c1 = self.linearc1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c1 = c1
        out = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        # out=self.expend(out)
        _c = self.linear_fuse(out)
        _c = self.gnconvM(_c)
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class FL_SegFormer(nn.Module):
    def __init__(self, num_classes=4, phi='b0', pretrained=False):
        super(SegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)


    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        x = self.backbone.forward(inputs)

        x = self.decode_head.forward(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

