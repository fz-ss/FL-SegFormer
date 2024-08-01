import numpy as np

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)
        self.smooth = 1e-5
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = (np.diag(self.confusionMatrix).sum()) / (self.confusionMatrix.sum()+self.smooth)
        return acc

    # def classPixelAccuracy(self):
    #     # return each category pixel accuracy(A more accurate way to call it precision)
    #     # acc = (TP) / TP + FP
    #     classAcc = (np.diag(self.confusionMatrix)) / (self.confusionMatrix.sum(axis=1)+self.smooth)
    #
    #
    #     return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    # def meanPixelAccuracy(self):
    #     classAcc = self.classPixelAccuracy()
    #     meanAcc = classAcc[1],classAcc[2],classAcc[3]
    #       # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
    #     return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def precision(self):
        pc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=1)+self.smooth)
        return pc

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = (intersection) / (union+self.smooth)  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        iou = self.IntersectionOverUnion()
        miou = np.nanmean(iou)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return miou
    def recall(self):
    #   recall=Tp/(tp+fn)

        w = np.diag(self.confusionMatrix)
        q = self.confusionMatrix.sum(axis=0)+self.smooth

        se = w/q
        return se

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))