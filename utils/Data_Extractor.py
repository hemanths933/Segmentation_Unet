import json
import os
import tensorflow as tf
import glob
import re
import numpy as np
import functools
import matplotlib.pyplot as plt

import cv2


class Data_Extractor:


    def __init__(self):
        dirname = os.path.dirname(__file__)
        print("In data extractor init")
        self.conf_path = os.path.join(os.path.abspath(os.path.join(dirname,os.pardir)), "config/Unet.conf")
        conf_file = open(self.conf_path,'r')
        conf_content = conf_file.read()
        configuration = json.loads(str(conf_content))

        self.train_images_path = configuration['train_images_path']
        self.train_labels_path =configuration['train_labels_path']
        self.test_images_path =configuration['test_images_path']
        self.test_labels_path =configuration['test_labels_path']
        conf_file.close()

    def _augment():
        print("augment")

    def process_pathnames(self,image_file,label_file):
        print("data extractor process pathnames()")
        img_str = tf.read_file(image_file)
        img = tf.image.decode_jpeg(img_str,channels=3)
        img = tf.cast(img,dtype=tf.float64)
        #image = tf.image.per_image_standardization(img)
        image = tf.divide(img,tf.constant(255,dtype=tf.float64))
        paddings = tf.constant([[30, 30, ], [30, 30], [0, 0]])
        padded_image = tf.pad(image, paddings, 'REFLECT')

        label_str = tf.read_file(label_file)
        lbl = tf.image.decode_jpeg(label_str, channels=1)
        lbl = tf.cast(lbl, dtype=tf.float64)
        lbl = tf.cast(tf.add(lbl,tf.constant(0.5,dtype=tf.float64)), tf.int64) # Thresholding operation
        lbl = tf.cast(lbl, dtype=tf.float64)
        #label = tf.image.per_image_standardization(lbl)
        label = tf.divide(lbl,tf.constant(255,dtype=tf.float64))
        label = tf.image.resize_images(label,[388,388])#,tf.image.ResizeMethod.BICUBIC)
        label = tf.cast(label, dtype=tf.float64)
        #label = tf.cast(label,tf.uint8)
        #lbl = tf.reshape(lbl,[-1,388,388,1])
        
        return padded_image,label

    def provide_data(self,dataset_query):

        #set path to the dataset
        if dataset_query is 'train':
            images_path = self.train_images_path
            labels_path = self.train_labels_path
        else:
            images_path = self.test_images_path
            labels_path = self.test_labels_path

        #get all filenames
        filenames = glob.glob(images_path+"/image*.jp*g")
        labels = glob.glob(labels_path+"/label*.jp*g")

        #check whether images and labels have valid names
        temp_filenames = []
        temp_labels = []
        for f in filenames:
            m = re.search('image_(.*).jp', f)
            if m:
                temp_filenames.append(m.group(1))
        for f in labels:
            m = re.search('label_(.*).jp', f)
            if m:
                temp_labels.append(m.group(1))
        if temp_filenames.sort()!=temp_labels.sort():
            raise Exception("Some filenames and labels dont match . Please rename the files properly")

        #check whether number of images is equal to number of labels
        if len(filenames)!=len(labels):
            raise Exception("The number of images and lables are different")
        print(filenames)
        print(labels)


        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(self.process_pathnames, num_parallel_calls=5)
        # if augment_flag:
        #     dataset = dataset.map(_augment, num_parallel_calls=5)
        dataset = dataset.shuffle(30)
        dataset = dataset.batch(batch_size=1).repeat()
        return dataset

if __name__ == '__main__':
    instance = Data_Extractor()
    data = instance.provide_data("train")
    iterator = data.make_one_shot_iterator()
    images,labels = iterator.get_next()
    print(images)
    print(labels)
    with tf.Session() as sess:
        imges = sess.run(images)
        lbles = sess.run(labels)

        #imges = np.expand_dims(imges, axis=0)
        print(imges)
        print(lbles)

        for i in range(1):
            cv2.imshow('image',imges[i])
            cv2.waitKey(1000)
            cv2.imshow('label', lbles[i])
            cv2.waitKey(1000)

