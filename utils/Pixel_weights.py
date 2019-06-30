import numpy as np
import cv2
import collections
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

class Pixel_weights:

    def __init__(self):
        print("pixel weights init")

    def compute_label_weights(self,image):
        weights = np.array(image)
        unique = np.unique(image)
        counter = collections.Counter(image.flatten())
        size = image.size
        for i in unique:
            weights[np.where(image==i)]= counter[i]
        weights = weights/size
        return weights

    def compute_pixel_weights(self,image_tensor):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            image = sess.run(image_tensor)
            image = np.reshape(image,[image.shape[1],image.shape[2]])
            # print("imageshape is ",image.shape)
            # plt.interactive(False)
            # plt.imshow(image)
            # plt.show()
            print('image is ',image)
        cnc = np.array(image == 1).astype(np.int8)
        (_, output, _, _) = cv2.connectedComponentsWithStats(cnc, 4, cv2.CV_32S)
        #print(output.shape)
        #print(np.max(output))
        maps = np.zeros([output.shape[0], output.shape[1], np.max(output)])
        for label in range(np.max(output)):
            maps[:, :, label] = cv2.distanceTransform(np.array(output != label).astype(np.uint8), 2, 5)
        #print(maps.shape)
        d = np.zeros([output.shape[0], output.shape[1]], dtype=np.int32)
        #print(d.shape)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                #print(np.sort(maps).shape)
                d[i, j] = 10 * np.exp(-np.square(np.sort(maps[i, j, :])[1] + np.sort(maps[i, j, :])[2]) / 200)

        weights = self.compute_label_weights(image)+d
        return weights


if  __name__=="__main__":
    obj = Pixel_weights()
    image = Image.open('B:/Tensorflow/Segmentation/UNet/data/ground-truth.jpg').convert('L')
    image = np.asarray(image)
    weights = obj.compute_pixel_weights(image)
    print(weights.shape)



