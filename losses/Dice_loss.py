from losses.Loss import Loss
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Dice_loss(Loss):
    def __init__(self):
        print("in Dice loss init")
        Loss.__init__(self)

    def dice_coeff(self,y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return score

    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.dice_coeff(y_true, y_pred)
        return loss

    def compute_loss(self,y, y_hat):
        loss = tf.losses.sigmoid_cross_entropy(y, y_hat) + self.dice_loss(y, y_hat)
        return loss

if __name__=='__main__':
    y = cv2.imread('B:/Tensorflow/Segmentation/UNet/data/train_labels/label_0.jpeg',0)
    y = y/255
    y = tf.convert_to_tensor(y,dtype=tf.float64)
    y = tf.reshape(y,[1,512,512,1])
    print(y.shape)

    y_hat = cv2.imread('B:/Tensorflow/Segmentation/UNet/data/train_labels/label_1.jpeg',0)
    y_hat = y_hat/255
    y_hat = tf.convert_to_tensor(y_hat,dtype=tf.float64)
    y_hat = tf.reshape(y_hat, [1, 512, 512, 1])
    print(y_hat.shape)

    loss = Dice_loss().compute_loss(y,y_hat)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(loss))
    #print(loss)