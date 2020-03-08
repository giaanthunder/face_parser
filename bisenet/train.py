import argparse
import datetime
import json
import traceback

import numpy as np
import tensorflow as tf
import data, models, loss, test

import os, shutil, sys, math, time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# ==============================================================================
# =                                    param                                   =
# ==============================================================================

parser = argparse.ArgumentParser()
# model
parser.add_argument('--epoch', dest='epoch', type=int, default=1, help='# of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=24)
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--gpu', dest='gpu', type=str, default='0', help='choose GPU to use. ie. --gpu 0,1')
parser.add_argument('--experiment_name', dest='experiment_name',
      default=datetime.datetime.now().strftime("test_no_%y%m%d_%H%M%S"))

args = parser.parse_args()

# training
epochs = args.epoch
batch_size = args.batch_size
lr_base = args.lr
experiment_name = args.experiment_name

# ==============================================================================
# =                                   init                                     =
# ==============================================================================
# save setting information
output_dir = './output/%s' % experiment_name
if not os.path.exists(output_dir):
   os.makedirs(output_dir, exist_ok=True)

with open('./output/%s/setting.txt' % experiment_name, 'w') as f:
   f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

# tạo chỗ save sample
sample_dir = './output/%s/%s' % (experiment_name,experiment_name)

if not os.path.exists(sample_dir):
   os.makedirs(sample_dir, exist_ok=True)

# ==============================================================================
# =                                    train                                   =
# ==============================================================================
# strategy
print("======= Create strategy =======")
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.experimental.CentralStorageStrategy()
print("Number of GPUs in use:", strategy.num_replicas_in_sync)



# optimizer
print("======= Create optimizers =======")
with strategy.scope():
   lr    = tf.Variable(initial_value=lr_base, trainable=False)
   opt = tf.optimizers.Adam(lr, beta_1 =0., beta_2=0.99)
   params= tf.Variable(initial_value=[0, 3], trainable=False, dtype=tf.int64)


# model
print("======= Create model_lst =======")
with strategy.scope():
   Bnet = models.BiSeNet(19)

def step(model_lst, inputs):
   def single_step(model_lst, inputs):
      Bnet = model_lst
      with tf.GradientTape() as tape:
         loss = losses.loss(model_lst, inputs)
         
      w = Bnet.trainable_variables
      grad = tape.gradient(loss, w)
      dw_w = zip(grad, w) 
      
      opt.apply_gradients(dw_w)
      
      return loss

   loss = strategy.experimental_run_v2(single_step, args=(model_lst, inputs,))
   loss = strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
   return loss

# ==============================================================================
# =                                    backup                                  =
# ==============================================================================
x = tf.ones([1, 512, 512, 3], dtype=tf.float32)

y = Bnet(x)

# tạo chỗ lưu tiến trình
checkpoint_dir = './output/%s/trained_model' % experiment_name
checkpoint = tf.train.Checkpoint(
   params=params,
   opt=opt,
   Bnet=Bnet 
)

# load checkpoint cũ
print("======= Load old save point =======")
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
   print("Restored from {}".format(manager.latest_checkpoint))
else:
   print("Initializing from scratch.")
start_ep, start_it = params.numpy()
# ==============================================================================
# =                                    train                                   =
# ==============================================================================


try:
   
   
   with strategy.scope():
      #tf.random.set_seed(np.random.randint(1 << 31))
      g_batch_size = batch_size * strategy.num_replicas_in_sync
      losses = loss.losses(g_batch_size, True) 
      tester = test.tester(sample_dir, batch_size, z_dim)
      
      # data
      print("======= Make data: %dx%d ======="%(img_size, img_size))
      tr_data = data.MaskCeleba(data_dir='/data2/01_luan_van/data/CelebAMask-HQ', 
            img_resize=img_size, batch_size=g_batch_size, part='train')
      tr_ite = iter(strategy.experimental_distribute_dataset(tr_data.ds))

   
   # create tf graph
   print("======= Create graph =======")
   graph = tf.function(step)

   # training loop
   print("======= Create training loop =======")
   it_per_epoch = tr_data.count // g_batch_size
   max_it = it_per_epoch * epochs
   
   for ep in range(start_ep, epochs):
      for it in range(start_it, it_per_epoch):
         tik = time.time()
         it_count = ep * it_per_epoch + it
         
         with strategy.scope():
            # update alpha
            lr.assign(lr_base / (10 ** (ep // 100)))
            
            # get 1 batch
            inputs = next(tr_ite)
            model_lst = (Bnet)

            # train G
            loss = step(model_lst, inputs)

            # save sample
            if it % 100 == 0:
               strategy.experimental_run_v2(tester.make_sample, args=(
                     model_lst, inputs, ep, it,))
            
         # progress display
         if it % 1 == 0:
            tok = time.time()
            duration  = tok-tik
            remain_it = max_it - it_count - 1
            time_in_sec = int(remain_it * duration)
            remain_h  = time_in_sec // 3600 
            remain_m  = time_in_sec % 3600 // 60
            remain_s  = time_in_sec % 60
            print("EPOCH %d/%d - Batch %d/%d, Time: %.3f s - Remain: %d batches, Estimate: %02d:%02d:%02d " \
                  % (ep, epochs-1, it, it_per_epoch-1, duration, remain_it, remain_h, remain_m, remain_s))
            print("loss: %.3f" % (loss.numpy()))

         # save model
         if it_count % 1000 == 0 and it_count != 0:
            params.assign([start_ep, start_it])
            manager.save()
            print('Model is saved!')
      start_it = 0
   start_ep = 0
   params.assign([start_ep, start_it])
   manager.save()
   print('Training finished...')
   print('Model is saved!')
# ==============================================================================
# =                                  save model                                =
# ==============================================================================
except:
   traceback.print_exc()
finally:
   params.assign([ep, it])
   manager.save()
   print('Emergency backup...')
   print('Model is saved!')







