import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
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
from utils.get_lr import (get_lr_scheduler, set_optimizer_lr, get_lr,weights_init)
def main():

    #  =========================================== parameters setting ==================================================
    #
    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='fl_segformer', type=str, help='type of model')
    parser.add_argument('--task', default='fl_segformer', help='task or dataset name')

    args = parser.parse_args()
    opt = get_config(args.task)  # please configure your hyper-parameter
    opt.save_path_code = "_"

    device = torch.device(opt.device)
    if opt.gray == "yes":
        from utils.utils_gray import JointTransform2D, ImageToImage2D
    else:
        from utils.utils_rgb import JointTransform2D, ImageToImage2D

    timestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
    boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + timestr
    if not os.path.isdir(boardpath):
        os.makedirs(boardpath)
    TensorWriter = SummaryWriter(boardpath)

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
    tf_train = JointTransform2D(crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0,
                                color_jitter_params=None, long_mask=True)  # image reprocessing
    tf_val = JointTransform2D(crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    train_dataset = ImageToImage2D(opt.data_path, opt.train_split, tf_train, opt.classes)
    val_dataset = ImageToImage2D(opt.data_path, opt.val_split, tf_val, opt.classes)  # return image, mask, and filename
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True)
    # train_dataset = ImageToImage2D(opt.data_path, opt.train_split, tf_train, opt.classes)
    # val_dataset = ImageToImage2D(opt.data_path, opt.val_split, tf_val, opt.classes)  # return image, mask, and filename
    # trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    # valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True)

    model = get_model(modelname=args.modelname, classes=opt.classes, phi=opt.phi, pretrained=opt.pretrained)
    weights_init(model)
    model.to(device)
    if opt.pre_trained:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(opt.load_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#

        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    # model.load_state_dict(torch.load(opt.load_path))
    # -------------------------------------------------------------------#
    #   判断当前batch_size，自适应调整学习率
    # -------------------------------------------------------------------#
    nbs = 16
    lr_limit_max = opt.learning_rate
    lr_limit_min = opt.lr_limit_min
    Init_lr_fit = min(max(opt.batch_size / nbs * opt.learning_rate, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(opt.batch_size / nbs * opt.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    criterion = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {}, weight_ce=1)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=Init_lr_fit, weight_decay=1e-5)
    # ---------------------------------------#
    #   获得学习率下降的公式
    # ---------------------------------------#
    lr_scheduler_func = get_lr_scheduler(opt.lr_decay_type, Init_lr_fit, Min_lr_fit, opt.epochs)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))


    with open(os.path.join(opt.data_path, "ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(opt.data_path, "ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    epoch_step = num_train // opt.batch_size
    epoch_step_val = num_val
    # if torch.cuda.device_count() > 1:  # distributed parallel training
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model, device_ids = [0,1]).cuda()

    #  ========================================== begin to train the model =============================================

    best_dice, loss_log = 0.0, np.zeros(opt.epochs+1)
    for epoch in range(opt.epochs):
        #  ------------------------------------ training ------------------------------------
        model.train()
        train_losses = 0
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        pbar = tqdm(total=epoch_step, desc=f'Epoch train {epoch + 1}/{opt.epochs}', postfix=dict, mininterval=0.3)
        for batch_idx, (input_image, ground_truth, *rest) in enumerate(trainloader):
            input_image = Variable(input_image.to(device=opt.device))
            ground_truth = Variable(ground_truth.to(device=opt.device))
            # ---------------------------------- forward ----------------------------------
            output = model(input_image).to(device=opt.device)
            train_loss = criterion(output, ground_truth)
            # ---------------------------------- backward ---------------------------------
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses += train_loss.item()
            pbar.set_postfix(**{'total_loss': train_losses / (batch_idx + 1),
                                'lr': get_lr(optimizer)
                                })
            pbar.update(1)
        pbar.close()
            # print(train_loss)
        #  ---------------------------- log the train progress ----------------------------

        TensorWriter.add_scalars('loss', {' train_losses':  train_losses/ (batch_idx + 1)}, epoch + 1)
        loss_log[epoch] = train_losses / (batch_idx + 1)
        #  ----------------------------------- evaluate -----------------------------------
        if (epoch+1) % opt.eval_freq == 0:
            pbar1 = tqdm(total=epoch_step_val, desc=f'Epoch eval {epoch + 1}/{opt.epochs}', postfix=dict,mininterval=0.3)
            pa, cpa, mf1, IoU, recall, f1, val_losses, mIou = get_eval(valloader, model, criterion, opt,
                                                                 pbar1)
            TensorWriter.add_scalars('loss', {'val_losses': val_losses}, epoch + 1)
            TensorWriter.add_scalars('f1', {'grainf1': f1[1], 'impurity0F1': f1[2], 'impurity1F1': f1[3]}, epoch + 1)
            TensorWriter.add_scalars('IoU', {'grainiou': IoU[1], 'impurity0Iou': IoU[2], 'impurity1Iou': IoU[3]},
                                     epoch + 1)
            TensorWriter.add_scalars('cpa', {'grainPa': cpa[1], 'impurity0Pa': cpa[2], 'impurity1Pa': cpa[3]},
                                     epoch + 1)
            TensorWriter.add_scalars('recall',{'graincall': recall[1], 'impurity0Call': recall[2], 'impurity1Call': recall[3]},
                                     epoch + 1)
            TensorWriter.add_scalar('mIou', mIou, epoch + 1)
            TensorWriter.add_scalar('mf1', mf1, epoch + 1)
            TensorWriter.add_scalar('pa', pa, epoch + 1)
            if mf1 > best_dice:
                best_dice = mf1
                timestr = time.strftime('%m%d%H%M')
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                save_path = opt.save_path + args.modelname + opt.save_path_code + 'weights'
                torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
        if epoch % opt.save_freq == 0 or epoch == (opt.epochs-1):
            if not os.path.isdir(opt.save_path):
                os.makedirs(opt.save_path)
            save_path = opt.save_path + args.modelname + opt.save_path_code + '_' + str(epoch)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    main()
            


