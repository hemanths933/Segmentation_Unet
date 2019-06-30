import numpy as np
from metrics.Metric import Metric
from PIL import Image
import matplotlib.pyplot as plt


class IOU(Metric):
    def __init__(self):
        print("in IOU init")
              
    def calculate(self,y_hat,y):
        print("y_hat shape is {} and y shape is {} ".format(y_hat.shape,y.shape))
        intersection = np.logical_and(y,y_hat)
        union = np.logical_or(y, y_hat)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

if __name__ == '__main__':
    image1 = Image.open('datasets/train/labels/label_0.jpeg').convert('L')
    image1 = np.asarray(image1)
    image2 = Image.open('datasets/train/labels/label_1.jpeg').convert('L')
    image2 = np.asarray(image2)
    metric = IOU()
    print("The IOU is ",metric.calculate(image1,image2))