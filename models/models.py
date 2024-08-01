from imp import IMP_HOOK
from .fl_segformer import FL_SegFormer
from .segformer import SegFormer
from .deeplab.deeplabv3_plus import DeepLab
from .UNet.unet import Unet
from .PSPNet.pspnet import PSPNet
from .HrNet.hrnet import HRnet





def get_model(modelname="Unet",  classes=4, phi="b0", pretrained=False):
    # elif modelname == "SETR_ConvFormer":
    #     model = Setr_ConvFormer(n_channels=img_channel, n_classes=classes, imgsize=img_size)
    if modelname == "fl_segformer":
        model = FL_SegFormer(num_classes=classes, phi=phi, pretrained=pretrained)
    elif modelname == "segformer1":
        model = SegFormer(num_classes=classes, phi=phi, pretrained=pretrained)
    elif modelname == "deeplab":
        model = DeepLab(num_classes=classes, backbone="mobilenet", downsample_factor=16, pretrained=False)
    elif modelname == "unet":
        model = Unet(num_classes=classes, pretrained=pretrained, backbone="vgg")
    elif modelname == "pspnet":
        model = PSPNet(num_classes=classes, backbone="resnet50", downsample_factor=16, pretrained=pretrained, aux_branch=False)
    elif modelname == "hrnet":
        model = HRnet(num_classes=classes, backbone="hrnetv2_w18", pretrained=pretrained)
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model