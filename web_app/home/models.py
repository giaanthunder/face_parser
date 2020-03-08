from django.db import models

# Create your models here.

# ==== tensorflow ====
import tensorflow as tf
import numpy as np
import os, sys, shutil
sys.path.append(os.path.abspath('..'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import bisenet
from face_alignment.face_alignment import image_align
from face_alignment.landmarks_detector import LandmarksDetector

# ==============================================================================
# =                                   STGAN                                    =
# ==============================================================================
class BnetGUI():
   def __init__(self):
      self.bnet = bisenet.models.pretrained_models()
      
   def align_img(self, ori_path, ali_path):
      detector = LandmarksDetector('shape_predictor_68_face_landmarks.dat')
      for i, face_landmarks in enumerate(detector.get_landmarks(ori_path), start=1):
         image_align(ori_path, ali_path, face_landmarks)

   def parse(self, ori_path, res_path):
      img = bisenet.data.load_img(ori_path)
      img_in = bisenet.data.preprocess(img, size=512)
      img_in = tf.expand_dims(img_in, axis=0)
      out, out16, out32 = self.bnet(img_in)
      label = out[0]
      img = bisenet.data.colorize(img, label)
      bisenet.data.to_file(img, res_path)


bnetgui = BnetGUI()
# ==============