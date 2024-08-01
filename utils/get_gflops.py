import time
import torch
from thop import profile
from models.deeplab.deeplabv3_plus import DeepLab
from models.UNet.unet import Unet# 这里按你自己模型的存放路径修改即可
from models.PSPNet.pspnet import PSPNet
from models.HrNet.hrnet import HRnet


def compute_gflops_and_model_size(model):
    input = torch.randn(1, 3, 256, 256).cuda()
    macs, params = profile(model, inputs=(input,), verbose=False)

    GFlops = macs * 2.0 / pow(10, 9)
    model_size = params * 4.0 / 1024 / 1024
    params_M = params / pow(10, 6)
    return params_M, model_size, GFlops


@torch.no_grad()
def compute_fps(model, shape, epoch=100, device=None):
    total_time = 0.0

    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for i in range(epoch):
        data = torch.randn(shape).cuda()

        start = time.time()
        outputs = model(data)
        end = time.time()

        total_time += (end - start)

    return total_time / epoch


def test_model_flops():
    model = HRnet()  # 这里使用你的模型
    model.cuda()

    params_M, model_size, gflops = compute_gflops_and_model_size(model)

    print('Number of parameters: {:.2f} M '.format(params_M))
    print('Size of model: {:.2f} MB'.format(model_size))
    print('Computational complexity: {:.2f} GFlops'.format(gflops))


def test_fps():
    model = HRnet()  # 这里使用你的模型
    model.cuda()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fps = compute_fps(model, (1, 3, 256, 256), device=device)
    print('device: {} - fps: {:.3f}s'.format(device.type, fps))


if __name__ == '__main__':
    test_model_flops()
    test_fps()