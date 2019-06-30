# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
from models.Model import Model
from losses.Pixelwise_weighted_loss import Pixelwise_weighted_loss
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
import math


class Unet(Model):
  def __init__(self):
    print("unet init")
    Model.__init__(self)
    self.learning_rate = 0.000001
    self.loss = Pixelwise_weighted_loss().compute_loss
    self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

  def crop(self,tensor, reference):
    # print(tensor.shape.as_list()[1]-reference.shape.as_list()[1])
    if ((tensor.shape.as_list()[1] - reference.shape.as_list()[1]) % 2 == 0):
      offset_x = (tensor.shape.as_list()[1] - reference.shape.as_list()[1]) // 2
    else:
      offset_x = ((tensor.shape.as_list()[1] - reference.shape.as_list()[1]) // 2) + 1

    # print(tensor.shape.as_list()[2]-reference.shape.as_list()[2])
    if ((tensor.shape.as_list()[2] - reference.shape.as_list()[2]) % 2 == 0):
      offset_y = (tensor.shape.as_list()[2] - reference.shape.as_list()[2]) // 2
    else:
      offset_y = ((tensor.shape.as_list()[2] - reference.shape.as_list()[2]) // 2) + 1

    offset = [0, offset_x, offset_y, 0]
    # print("offset is ",offset)
    cropped_tensor = tf.slice(tensor, offset, [-1, reference.shape.as_list()[1], reference.shape.as_list()[2], -1])

    return cropped_tensor


  def concat(self,tensor,reference):
    cropped = self.crop(tensor,reference)
    return tf.concat([cropped,reference],axis=-1)

  def network(self,images):
    input_layer = tf.reshape(images,[-1,572,572,3])
    print(input_layer)

    W1 = tf.get_variable(name='W1',shape=[3,3,3,64],initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    conv1 = tf.nn.conv2d(input_layer,W1,padding="VALID",strides=[1,1,1,1])
    conv1 = tf.nn.relu(conv1)
    #conv1 = tf.layers.conv2d(inputs=input_layer,filters=W1,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)
    print(conv1)

    W2 = tf.get_variable(name='W2', shape=[3, 3, 64, 64], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    conv2 = tf.nn.conv2d(conv1,W2,padding="VALID",strides=[1,1,1,1])
    conv2 = tf.nn.relu(conv2)
    #conv2 = tf.layers.conv2d(inputs=conv1,filters=W2,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)#,name = 'conv_merge_4'
    print(conv2)
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    print(pool1)

    W3 = tf.get_variable(name='W3', shape=[3, 3, 64, 128], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    conv3 = tf.nn.conv2d(pool1,W3,padding="VALID",strides=[1,1,1,1])
    conv3 = tf.nn.relu(conv3)
    #conv3 = tf.layers.conv2d(inputs=pool1,filters=W3,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)
    print(conv3)

    W4 = tf.get_variable(name='W4', shape=[3, 3, 128, 128], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    conv4 = tf.nn.conv2d(conv3,W4,padding="VALID",strides=[1,1,1,1])
    conv4 = tf.nn.relu(conv4)
    #conv4 = tf.layers.conv2d(inputs=conv3,filters=W4,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)#,name = 'conv_merge_3'
    print(conv4)
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    print(pool2)

    W5 = tf.get_variable(name='W5', shape=[3, 3, 128, 256], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    conv5 = tf.nn.conv2d(pool2,W5,padding="VALID",strides=[1,1,1,1])
    conv5 = tf.nn.relu(conv5)
    #conv5 = tf.layers.conv2d(inputs=pool2,filters=W5,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)
    print(conv5)

    W6 = tf.get_variable(name='W6', shape=[3, 3, 256, 256], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    conv6 = tf.nn.conv2d(conv5,W6,padding="VALID",strides=[1,1,1,1])
    conv6 = tf.nn.relu(conv6)
    #conv6 = tf.layers.conv2d(inputs=conv5,filters=W6,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)#,name = 'conv_merge_2'
    print(conv6)
    pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)
    print(pool3)

    W7 = tf.get_variable(name='W7', shape=[3, 3, 256, 512], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    conv7 = tf.nn.conv2d(pool3,W7,padding="VALID",strides=[1,1,1,1])
    conv7 = tf.nn.relu(conv7)
    #conv7 = tf.layers.conv2d(inputs=pool3,filters=W7,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)
    print(conv7)

    W8 = tf.get_variable(name='W8', shape=[3, 3, 512, 512], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    conv8 = tf.nn.conv2d(conv7,W8,padding="VALID",strides=[1,1,1,1])
    conv8 = tf.nn.relu(conv8)
    #conv8 = tf.layers.conv2d(inputs=conv7,filters=W8,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)#,name = 'conv_merge_1'
    print(conv8)
    pool4 = tf.layers.max_pooling2d(inputs=conv8, pool_size=[2, 2], strides=2,name = 'p4')
    print(pool4)

    W9 = tf.get_variable(name='W9', shape=[3, 3, 512, 1024], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    conv9 = tf.nn.conv2d(pool4,W9,padding="VALID",strides=[1,1,1,1])
    conv9 = tf.nn.relu(conv9)
    #conv9 = tf.layers.conv2d(inputs=pool4,filters=W9,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)
    print(conv9)

    W10 = tf.get_variable(name='W10', shape=[3, 3, 1024, 1024], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    conv10 = tf.nn.conv2d(conv9,W10,padding="VALID",strides=[1,1,1,1])
    conv10 = tf.nn.relu(conv10)
    #conv10 = tf.layers.conv2d(inputs=conv9,filters=W10,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)
    print(conv10)
    W11 = tf.get_variable(name='W11', shape=[2, 2, 512, 1024], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    deconv1 = tf.nn.conv2d_transpose(conv10,W11,strides = [1,2,2,1],padding='VALID',output_shape=[1,56,56,512])
    deconv1 = tf.nn.relu(deconv1)
    #deconv1 = tf.layers.conv2d_transpose(inputs=conv10,filters=W11,kernel_size=[2, 2],strides = (2,2),padding='valid',activation=tf.nn.relu)
    print(deconv1)

    concat1 = self.concat(conv8,deconv1)
    print("concat1:",concat1)
    W12 = tf.get_variable(name='W12', shape=[3, 3, 1024, 512], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    conv11 = tf.nn.conv2d(concat1,W12,padding="VALID",strides=[1,1,1,1])
    conv11 = tf.nn.relu(conv11)
    #conv11 = tf.layers.conv2d(inputs=concat1,filters=W12,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)
    print(conv11)

    W13 = tf.get_variable(name='W13', shape=[3, 3, 512, 512], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    conv12 = tf.nn.conv2d(conv11,W13,padding="VALID",strides=[1,1,1,1])
    conv12 = tf.nn.relu(conv12)
    #conv12 = tf.layers.conv2d(inputs=conv11,filters=W13,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)
    print(conv12)

    W14 = tf.get_variable(name='W14', shape=[2, 2, 256, 512], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    deconv2 = tf.nn.conv2d_transpose(conv12,W14,strides = [1,2,2,1],padding='VALID',output_shape=[1,104,104,256])
    deconv2 = tf.nn.relu(deconv2)
    #deconv2 = tf.layers.conv2d_transpose(inputs=conv12,filters=W14,kernel_size=[2, 2],strides = (2,2),padding='valid',activation=tf.nn.relu)
    print(deconv2)

    concat2 = self.concat(conv6,deconv2)
    print("concat2:",concat2)
    W15 = tf.get_variable(name='W15', shape=[3, 3, 512, 256], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    conv13 = tf.nn.conv2d(concat2,W15,padding="VALID",strides=[1,1,1,1])
    conv13 = tf.nn.relu(conv13)
    #conv13 = tf.layers.conv2d(inputs=concat2,filters=W15,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)
    print(conv13)

    W16 = tf.get_variable(name='W16', shape=[3, 3, 256, 256], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    conv14 = tf.nn.conv2d(conv13,W16,padding="VALID",strides=[1,1,1,1])
    conv14 = tf.nn.relu(conv14)
    #conv14 = tf.layers.conv2d(inputs=conv13,filters=W16,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)
    print(conv14)

    W17 = tf.get_variable(name='W17', shape=[2, 2, 128, 256], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    deconv3 = tf.nn.conv2d_transpose(conv14,W17,strides = [1,2,2,1],padding='VALID',output_shape=[1,200,200,128])
    deconv3 = tf.nn.relu(deconv3)
    #deconv3 = tf.layers.conv2d_transpose(inputs=conv14,filters=W17,kernel_size=[2, 2],strides = (2,2),padding='valid',activation=tf.nn.relu)
    print(deconv3)

    concat3 = self.concat(conv4,deconv3)
    print("concat3:",concat3)
    W18 = tf.get_variable(name='W18', shape=[3, 3, 256, 128], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    conv15 = tf.nn.conv2d(concat3,W18,padding="VALID",strides=[1,1,1,1])
    conv15 = tf.nn.relu(conv15)
    #conv15 = tf.layers.conv2d(inputs=concat3,filters=W18,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)
    print(conv15)

    W19 = tf.get_variable(name='W19', shape=[3, 3, 128, 128], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    conv16 = tf.nn.conv2d(conv15,W19,padding="VALID",strides=[1,1,1,1])
    conv16 = tf.nn.relu(conv16)
    #conv16 = tf.layers.conv2d(inputs=conv15,filters=W19,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)
    print(conv16)

    W20 = tf.get_variable(name='W20', shape=[2, 2, 64, 128], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    deconv4 = tf.nn.conv2d_transpose(conv16,W20,strides = [1,2,2,1],padding='VALID',output_shape=[1,392,392,64])
    deconv4 = tf.nn.relu(deconv4)
    #deconv4 = tf.layers.conv2d_transpose(inputs=conv16,filters=W20,kernel_size=[2, 2],strides = (2,2),padding='valid',activation=tf.nn.relu)
    print(deconv4)

    concat4 = self.concat(conv2,deconv4)
    print("concat4:",concat4)
    W21 = tf.get_variable(name='W21', shape=[3, 3, 128, 64], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    conv17 = tf.nn.conv2d(concat4,W21,padding="VALID",strides=[1,1,1,1])
    conv17 = tf.nn.relu(conv17)
    #conv17 = tf.layers.conv2d(inputs=concat4,filters=W21,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)
    print(conv17)

    W22 = tf.get_variable(name='W22', shape=[3, 3, 64, 64], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    conv18 = tf.nn.conv2d(conv17,W22,padding="VALID",strides=[1,1,1,1])
    conv18 = tf.nn.relu(conv18)
    #conv18 = tf.layers.conv2d(inputs=conv17,filters=W22,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)
    print(conv18)

    W23 = tf.get_variable(name='W23', shape=[1, 1, 64, 1], initializer=tf.initializers.random_normal(mean=0,stddev=math.sqrt(2/576)))
    output = tf.nn.conv2d(conv18,W23,padding="VALID",strides=[1,1,1,1])

    #output = tf.layers.conv2d(inputs=conv18,filters=W23,kernel_size=[1, 1],padding="valid",activation=tf.nn.sigmoid)
    print(output)

    return output

  def network_keras(self):
    input_layer = layers.Input(shape=[572,572,3])

    conv1 = layers.Conv2D(filters=64,kernel_size=[3, 3],padding="valid",activation=tf.nn.relu)(input_layer)
    conv2 = layers.Conv2D(filters=64, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)(conv1)
    pool1 = layers.MaxPool2D(pool_size=[2, 2], strides=2)(conv2)

    conv3 = layers.Conv2D(filters=128, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)(pool1)
    conv4 = layers.Conv2D(filters=128, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)(conv3)
    pool2 = layers.MaxPool2D(pool_size=[2, 2], strides=2)(conv4)

    conv5 = layers.Conv2D(filters=256, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)(pool2)
    conv6 = layers.Conv2D(filters=256, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)(conv5)
    pool3 = layers.MaxPool2D(pool_size=[2, 2], strides=2)(conv6)

    conv7 = layers.Conv2D(filters=512, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)(pool3)
    conv8 = layers.Conv2D(filters=512, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)(conv7)
    pool4 = layers.MaxPool2D(pool_size=[2, 2], strides=2)(conv8)

    conv9 = layers.Conv2D(filters=1024, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)(pool4)
    conv10 = layers.Conv2D(filters=1024, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)(conv9)
    deconv1 = layers.Conv2DTranspose(filters=512,kernel_size=[2, 2],strides = (2,2),padding='valid',activation=tf.nn.relu)(conv10)

    concat1 = layers.Lambda(self.concat,arguments={'reference':deconv1})(conv8)
    #concat1 = self.concat(conv8, deconv1)
    conv11 = layers.Conv2D(filters=512, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)(concat1)
    conv12 = layers.Conv2D(filters=512, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)(conv11)
    deconv2 = layers.Conv2DTranspose(filters=256, kernel_size=[2, 2], strides=(2, 2), padding='valid',activation=tf.nn.relu)(conv12)

    concat2 = layers.Lambda(self.concat,arguments={'reference':deconv2})(conv6)
    #concat2 = self.concat(conv6, deconv2)
    conv13 = layers.Conv2D(filters=256, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)(concat2)
    conv14 = layers.Conv2D(filters=256, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)(conv13)
    deconv3 = layers.Conv2DTranspose(filters=128, kernel_size=[2, 2], strides=(2, 2), padding='valid',activation=tf.nn.relu)(conv14)

    concat3 = layers.Lambda(self.concat,arguments={'reference':deconv3})(conv4)
    #concat3 = self.concat(conv4, deconv3)
    conv15 = layers.Conv2D(filters=128, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)(concat3)
    conv16 = layers.Conv2D(filters=128, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)(conv15)
    deconv4 = layers.Conv2DTranspose(filters=64, kernel_size=[2, 2], strides=(2, 2), padding='valid',
                                     activation=tf.nn.relu)(conv16)

    concat4 = layers.Lambda(self.concat,arguments={'reference':deconv4})(conv2)
    #concat4 = self.concat(conv2, deconv4)
    conv17 = layers.Conv2D(filters=64, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)(concat4)
    conv18 = layers.Conv2D(filters=64, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu)(conv17)
    output_layer = layers.Conv2D(filters=1, kernel_size=[1, 1], padding="valid", activation=tf.nn.sigmoid)(conv18)


    model = models.Model(inputs = input_layer,outputs=output_layer)
    return model


if __name__=='__main__':
  net = Unet()
  net.train()