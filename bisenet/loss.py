from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python import keras
import math, sys, os


class losses():
   def __init__(self, g_batch_size = 1, dist_train=False):
      self.dist_train   = dist_train
      self.g_batch_size = g_batch_size
      
   def loss(self, model_lst, inputs):
      model = model_lst
      y_true, y_pred = inputs
      loss = 0
      return loss
      
      
   # ==============================================================================
   # =                                  Utilities                                 =
   # ==============================================================================
   def mean(self, loss):
      if self.dist_train:
         return  tf.reduce_sum(loss) * (1. / self.g_batch_size)
      else:
         return tf.reduce_mean(loss)
#








