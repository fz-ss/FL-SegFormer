
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import torch.nn.functional as F
import utils.metrics as metrics
from utils.record_tools import get_records, get_records2
from hausdorff import hausdorff_distance
from utils.visualization import visual_segmentation
from utils.SegmentationMetric import SegmentationMetric
def eval_2d_slice(valloader, model, criterion, opt,pbar1):
    model.eval()

    val_losses, pa, mf1, mIoU = 0, 0, 0, 0
    cpa = np.zeros(opt.classes)
    recall = np.zeros(opt.classes)
    IoU, f1 = np.zeros(opt.classes), np.zeros(opt.classes)
    for batch_idx, (input_image, ground_truth, *rest) in enumerate(valloader):
        input_image = Variable(input_image.to(device=opt.device))
        ground_truth = Variable(ground_truth.to(device=opt.device))
        if isinstance(rest[0][0], str):
            image_filename = rest[0][0]
        else:
            image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
        with torch.no_grad():
            predict = model(input_image)

        val_loss = criterion(predict, ground_truth)
        val_losses += val_loss.item()

        gt = ground_truth.detach().cpu().numpy()
        predict = F.softmax(predict, dim=1)
        pred = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = np.argmax(pred, axis=1)  # (b, h, w)
        b, h, w = seg.shape
        metric = SegmentationMetric(4)
        metric.addBatch(seg, gt)
        #  IoU, cpa, recall, f1, recall
        pa += metric.pixelAccuracy()
        cpa += metric.precision()

        recall += metric.recall()
        IoU += metric.IntersectionOverUnion()
        mIoU += metric.meanIntersectionOverUnion()
        f1 += 2 * (metric.recall() * metric.precision()) / (metric.recall() + metric.precision()+opt.smooth)
        mf1 += np.nanmean(2 * (metric.recall() * metric.precision()) / (metric.recall() + metric.precision()+opt.smooth))

        pbar1.set_postfix(**{'val_losses': val_losses / (batch_idx + 1),
                             'pa': pa / (batch_idx + 1),
                             'mf1': mf1 / (batch_idx + 1),
                             'mIoU': mIoU / (batch_idx + 1)})

        # for i in range(0, opt.classes):
        #     pred_i = np.zeros((b, h, w))
        #     pred_i[seg == i] = 255
        #     gt_i = np.zeros((b, h, w))
        #     gt_i[gt == i] = 255
        #     dices[i] += metrics.dice_coefficient(pred_i, gt_i)
        #     iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
        #     ious[i] += iou
        #     accs[i] += acc
        #     ses[i] += se
        #     sps[i] += sp
        #     hds[i] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
        #     del pred_i, gt_i
        if opt.visual:
            visual_segmentation(seg, image_filename, opt)
        pbar1.update(1)
    pa = pa / (batch_idx + 1)
    cpa = cpa / (batch_idx + 1)
    mf1 = mf1 / (batch_idx + 1)
    IoU = IoU/(batch_idx + 1)
    mIoU = mIoU / (batch_idx + 1)
    recall = recall/(batch_idx + 1)
    f1 = f1/(batch_idx + 1)
    val_losses = val_losses / (batch_idx + 1)

    #return dices, mean_dice, val_losses
    if opt.mode == "train":
        pbar1.close()
        return pa, cpa, mf1, IoU, recall, f1, val_losses, mIoU
    else:
        pbar1.close()
        return pa, cpa, mf1, IoU, recall, f1, val_losses, mIoU






def get_eval(valloader, model, criterion, opt, pbar1):
    if opt.eval_mode == "slice":
        return eval_2d_slice(valloader, model, criterion, opt,  pbar1)
    # elif opt.eval_mode == "slice_record":
    #     return eval_slice_record(valloader, model, criterion, opt)
    # elif opt.eval_mode == "slice_visual":
    #     return eval_slice_visual(valloader, model, criterion, opt)
    # elif opt.eval_mode == "patient":
    #     return eval_2d_patient(valloader, model, criterion, opt)
    # elif opt.eval_mode == "patient_record":
    #     return eval_patient_record(valloader, model, criterion, opt)
    else:
        raise RuntimeError("Could not find the eval mode:", opt.eval_mode)