from __future__ import absolute_import, division, print_function

import argparse
import datetime
import json
import traceback

import numpy as np
import tensorflow as tf

import data, models, loss, test

import os, shutil, sys, math, time
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


class tester():
   def __init__(self, sample_dir, batch_size, z_dim=512):
      self.sample_dir = sample_dir
      self.batch_size = batch_size
      self.z_dim      = z_dim

   def make_sample(self, model_lst, inputs, epoch, batch, samp_per_batch=3, att_name="fake"):
      x_real, x_fake, x_rec = self.create_sample_data(model_lst, inputs)
      
      for i in range(samp_per_batch):
         img_name = self.sample_dir+"/sample_%dx%d_%03d_%03d_%d_"%(
               cur_res, cur_res,epoch, batch, i)
         # real
         im = self.convert_to_img(x_real[i])
         im.save(img_name+"real.jpg")
         
         # fake
         im = self.convert_to_img(x_fake[i])
         im.save(img_name+att_name+".jpg")
         
         # real recovery
         im = self.convert_to_img(x_rec[i])
         im.save(img_name+"rec.jpg")

   def create_sample_data(self, model_lst, inputs):
      Gen, Dis, Enc, Stu = model_lst
      x_real, a, b = inputs
      b_diff = b - a
      a_diff = a - a

      z_real     = Enc(x_real)

      return (x_real, x_fake, x_rec)

   def convert_to_img(self, x):
      img = ( (x.numpy()+1)*127.5 ).astype('uint8')
      return Image.fromarray(img)
   
if __name__ == '__main__':
   # ==============================================================================
   # =                                    param                                   =
   # ==============================================================================

 
   Gen = models.Generator()

   manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
   checkpoint.restore(manager.latest_checkpoint)
   if manager.latest_checkpoint:
      print("Restore success from:")
      print(manager.latest_checkpoint)
   else:
      print("Restore fail")
   
   test_count, test_data = data.img_ds(data_dir='/data2/01_luan_van/data/', 
      atts=atts, img_resize=img_size, batch_size=batch_size, part='test')
   test_ite = iter(test_data)

   it_per_epoch = test_count//batch_size
   for it in range(it_per_epoch):
      model_lst = (Gen, Dis, Enc, Stu)
      x_real, a = next(test_ite)
      #np.save("/data2/01_luan_van/img_test.npy", x_real.numpy())
      x_real = tf.convert_to_tensor(np.load("/data2/01_luan_van/img_test.npy"))
      a = (a * 2 - 1) * 0.5
      

 #
 
 
 
 
 
 