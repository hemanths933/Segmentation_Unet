from abc import abstractproperty,abstractmethod,ABC
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import layers
from utils.Data_Extractor import Data_Extractor
import matplotlib.pyplot as plt
from tensorflow.python.keras import models
import tensorflow.python.keras

import os

class Model(ABC):
  
  def __init__(self):
    self.data_extractor = Data_Extractor()


  @abstractmethod
  def network(self,images):
    pass
  
  def train_keras(self):
    print("Model train")
    self.data = self.data_extractor.provide_data('train')

    checkpoint_path = 'trained-models/unet.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,save_weights_only=True,verbose=1)

    # iterator = self.data.make_one_shot_iterator()
    # images, actual_output = iterator.get_next()
    # images = tf.cast(images, tf.float32)


    # actual_output = tf.cast(actual_output, tf.float32)
    # print("actual output size is ",actual_output)

    model = self.network_keras()
    model.compile(optimizer=self.optimizer,loss=self.loss,metrics=['accuracy'])
    model.fit(self.data,steps_per_epoch=1,epochs=100,callbacks=[cp_callback])
    # with tf.Session() as sess:
    #   sess.run(tf.global_variables_initializer())
    #   for i in range(1000):
    #
    #     l,y_hat,y,_ = sess.run([loss,prediction,actual_output,optimize])
    #     print("loss is ",l)

  def train(self):

    checkpoint_path = 'trained-models/unet.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    print("Model train")
    self.data = self.data_extractor.provide_data('train')

    iterator = self.data.make_one_shot_iterator()
    images, actual_output = iterator.get_next()
    images = tf.cast(images, tf.float32)

    actual_output = tf.cast(actual_output, tf.float32)
    print("actual output size is ", actual_output)

    prediction = self.network(images)
    loss = self.loss(actual_output, prediction)  # loss must have softmax included with the loss function
    optimizer = self.optimizer
    optimize = optimizer.minimize(loss)
    loss_summary = tf.summary.scalar('loss',loss)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    #builder = tf.saved_model.builder.SavedModelBuilder(checkpoint_path)
    with tf.Session() as sess:
      sum_writer = tf.summary.FileWriter('C:/Users/hemanth sharma/PycharmProjects/Unet/visualization', sess.graph)
      sess.run(tf.global_variables_initializer())

      for epoch in range(1000):
        summary = sess.run(loss_summary)
        sum_writer.add_summary(summary,epoch)
        l, y_hat, y, _ = sess.run([loss, prediction, actual_output, optimize])
        print("loss is ", l)
        saver.save(sess,checkpoint_path)
        #builder.save()

  def test(self):
    print("Model test")
    self.data = self.data_extractor.provide_data('test')

    iterator = self.data.make_one_shot_iterator()
    images, actual_output = iterator.get_next()
    images = tf.cast(images, tf.float32)
    prediction = self.network(images)
    saver = tf.train.Saver()
    with tf.Session() as sess:

      saver.restore(sess,'trained-models/unet.ckpt')
      #tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], 'trained-models/unet.ckpt')
      for example in range(30):
        pred,out = sess.run([prediction,actual_output])
        for i in range(pred.shape[0]):
          plt.imshow(pred.reshape(pred.shape[1],pred.shape[2]))
          plt.show()
        for i in range(out.shape[0]):
          plt.imshow(out.reshape(out.shape[1],out.shape[2]))
          plt.show()
      # print("Prediction is ",pred)
      # print("Actual output is ",out)


  def evaluate(self):
    print("Model evaluate")
#     metric.calculate(y_hat_test,label_y_hat_test)

