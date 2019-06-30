import numpy as np
from losses.Loss import Loss
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.Pixel_weights import Pixel_weights
from PIL import Image
import cv2

class Pixelwise_weighted_loss(Loss):

    def __init__(self):
        print("pixel wise weighted loss init")
        Loss.__init__(self)

    def compute_loss(self,y, y_hat):
        weights = Pixel_weights().compute_pixel_weights(y)
        weights = weights/np.max(weights)
        weights = np.expand_dims(weights,axis=0)
        print("weights dimension is ",weights.shape)
        print("max and min value in weights {} ".format(np.max(weights)))
        print("y is ",y.shape)
        print("y_hat is ",y_hat.shape)

        w = tf.constant(weights)
        y = tf.reshape(y,[-1,y.shape[1],y.shape[2]])
        y_hat = tf.reshape(y_hat,[-1,y_hat.shape[1],y_hat.shape[2]])
        # print("altered y shape is ", y.shape)
        # print("altered y_hat shape is ", y_hat.shape)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            image1,image2 = sess.run([y,y_hat])
            print('image1 max is',np.max(image1))
            print('image2 max is',np.max(image2))
            plt.imshow(image1.reshape(image1.shape[1],image1.shape[2]))
            plt.show()
            plt.imshow(image2.reshape(image2.shape[1],image2.shape[2]))
            plt.show()
        loss = tf.losses.sigmoid_cross_entropy(y,y_hat,weights=w)
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     l = sess.run(loss)
        # print("loss is ",l)
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

    loss = Pixelwise_weighted_loss().compute_loss(y,y_hat)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(loss))
    #print(loss)