import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.loss_functions.dice_loss import DC_and_BCE_loss, DC_and_CE_loss, SoftDiceLoss
from utils.config import get_config
from models.models import get_model
from utils.evaluation import get_eval
from tqdm import tqdm

def main():

    #  =========================================== parameters setting ==================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='segformer', type=str, help='type of model')
    parser.add_argument('--task', default='segformer', help='task or dataset name')

    args = parser.parse_args()
    opt = get_config(args.task)  # please configure your hyper-parameter
    #opt.eval_mode = "patient_record"
    opt.save_path_code = "_"
    opt.mode = "eval"
    opt.visual = True
    print(opt.load_path)

    device = torch.device(opt.device)
    if opt.gray == "yes":
        from utils.utils_gray import JointTransform2D, ImageToImage2D
    else:
        from utils.utils_rgb import JointTransform2D, ImageToImage2D

    # torch.backends.cudnn.enabled = True # Whether to use nondeterministic algorithms to optimize operating efficiency
    # torch.backends.cudnn.benchmark = True

    #  ============================= add the seed to make sure the results are reproducible ============================

    seed_value = 30  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  ============================================= model initialization ==============================================
    tf_test = JointTransform2D(crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    test_dataset = ImageToImage2D(opt.data_path, opt.test_split, tf_test, opt.classes)  # return image, mask, and filename
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    with open(os.path.join(opt.data_path, "ImageSets/Segmentation/test.txt"), "r") as f:
        val_lines = f.readlines()
    num_val = len(val_lines)
    epoch_step_val = num_val
    pbar1 = tqdm(total=epoch_step_val, desc=f'test', postfix=dict, mininterval=0.3)
    model = get_model(modelname=args.modelname, classes=opt.classes, phi=opt.phi, pretrained=opt.pretrained)
    model.to(device)
    model.load_state_dict(torch.load(opt.load_path))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    criterion = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {}, weight_ce=1)
    #dices, mean_dice, val_losses, rtoken1, rtoken2, rtoken3, rmap1, rmap2, rmap3 = get_eval(testloader, model, criterion, opt)
    #dices, mean_dice, val_losses = get_eval(testloader, model, criterion, opt)
    if opt.mode == "train":
        pa, cpa, mf1, IoU, recall, f1, val_losses, mIou = get_eval(testloader, model, criterion, opt, pbar1)
        print( pa, cpa, mf1, IoU, recall, f1, val_losses, mIou)
    else:
        pa, cpa, mf1, IoU, recall, f1, val_losses, mIou = get_eval(testloader, model, criterion, opt, pbar1)
        print( pa, cpa, mf1, IoU, recall, f1, val_losses, mIou)



if __name__ == '__main__':
    main()
            


