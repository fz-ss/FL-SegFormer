import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import random
from utils.config import get_config
from models.models import get_model
import torch.nn.functional as F
from PIL import Image
import torch

def main():
    #  =========================================== parameters setting ==================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='segformer', type=str, help='type of model')
    parser.add_argument('--task', default='segformer', help='task or dataset name')

    args = parser.parse_args()
    opt = get_config(args.task)  # please configure your hyper-parameter
    # opt.eval_mode = "patient_record"
    opt.save_path_code = "_"
    opt.mode = "eval"
    opt.visual = True
    print(opt.load_path)

    device = torch.device(opt.device)
    if opt.gray == "yes":
        from utils.utils_gray import JointTransform2D, ImageToImage2D
    else:
        from utils.utils_rgb import JointTransform2D, ImageToImage2D



    seed_value = 30  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  ============================================= model initialization ==============================================
    model = get_model(modelname=args.modelname, classes=opt.classes, phi=opt.phi, pretrained=opt.pretrained)
    model.to(device)
    model.load_state_dict(torch.load(opt.load_path))

    img = 'C:/Users/yhz/Desktop/ConvFormer-main/data/拼接/1/img84_13.jpg'
    image = Image.open(img)
    image = image.convert('RGB')
    image = np.expand_dims(np.transpose(np.array(image, np.float32), (2, 0, 1)), 0)
    image = torch.from_numpy(image)
    image = image.cuda()

    with torch.no_grad():
        model.eval()
        input = model.backbone(image)
        c1, c2, c3, c4 = model.decode_head.fpn(input)
        _c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c1 = c1
        out = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        _c = model.decode_head.linear_fuse(out)
        _c = model.decode_head.gnconvM(_c)

        _c = F.interpolate(_c, size=(256,256), mode='bilinear', align_corners=False)
        _c = _c.squeeze()
        hot_data = torch.sum(_c.cpu(), dim=0).squeeze()
        print(hot_data.size())
        torch.save(hot_data, 'tensor.pt')
        a = torch.load('tensor.pt')
        import matplotlib.pyplot as plt
        plt.imshow(a, cmap=plt.cm.jet)

        plt.axis('off')  # Turn off axis

        # Save the resized image with transparent background
        plt.savefig('C:/Users/yhz/Desktop/ConvFormer-main/data/img84_13.png', transparent=True,
                    bbox_inches='tight', pad_inches=0)

        plt.show()


if __name__ == '__main__':
    main()



