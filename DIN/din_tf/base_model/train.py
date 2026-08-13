import os
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import time
import pickle
import random
import numpy as np
import sys
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

from input import DataInput, DataInputTest
from model import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
random.seed(1234)
np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

train_batch_size = 32
test_batch_size = 512

with open('dataset.pkl', 'rb') as f:
  train_set = pickle.load(f)
  test_set = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count = pickle.load(f)

best_auc = 0.0
def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d:d[2])

    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None

def _auc_arr(score):
  score_p = score[:,0]
  score_n = score[:,1]
  #print "============== p ============="
  #print score_p
  #print "============== n ============="
  #print score_n
  score_arr = []
  for s in score_p.tolist():
    score_arr.append([0, 1, s])
  for s in score_n.tolist():
    score_arr.append([1, 0, s])
  return score_arr
def _eval(sess, model):
  auc_sum = 0.0
  score_arr = []
  for _, uij in DataInputTest(test_set, test_batch_size):
    auc_, score_ = model.eval(sess, uij)
    score_arr += _auc_arr(score_)
    auc_sum += auc_ * len(uij[0])
  test_gauc = auc_sum / len(test_set)
  Auc = calc_auc(score_arr)
  global best_auc
  if best_auc < test_gauc:
    best_auc = test_gauc
    model.save(sess, 'save_path/ckpt')
  return test_gauc, Auc


gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
with tf.compat.v1.Session(config=config) as sess:

  print('=== TF Environment Info ===')
  print('TF version:', tf.__version__)
  print('Built with CUDA:', tf.test.is_built_with_cuda())
  gpu_name = tf.test.gpu_device_name()
  print('GPU device name:', gpu_name if gpu_name else '(NO GPU DETECTED — running on CPU)')
  physical_gpus = tf.config.list_physical_devices('GPU')
  print('Physical GPUs:', physical_gpus if physical_gpus else '(none)')
  for d in sess.list_devices():
    print('  -', d.name, d.device_type)
  print('===========================')
  sys.stdout.flush()

  model = Model(user_count, item_count, cate_count, cate_list)
  sess.run(tf.compat.v1.global_variables_initializer())
  sess.run(tf.compat.v1.local_variables_initializer())

  print('test_gauc: %.4f\t test_auc: %.4f' % _eval(sess, model))
  sys.stdout.flush()
  lr = 1.0
  start_time = time.time()
  for _ in range(50):

    random.shuffle(train_set)

    epoch_size = round(len(train_set) / train_batch_size)
    loss_sum = 0.0
    for _, uij in DataInput(train_set, train_batch_size):
      loss = model.train(sess, uij, lr)
      loss_sum += loss

      if model.global_step.eval() % 1000 == 0:
        test_gauc, Auc = _eval(sess, model)
        print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
              (model.global_epoch_step.eval(), model.global_step.eval(),
               loss_sum / 1000, test_gauc, Auc))
        sys.stdout.flush()
        loss_sum = 0.0

      if model.global_step.eval() % 336000 == 0:
        lr = 0.1

    print('Epoch %d DONE\tCost time: %.2f' %
          (model.global_epoch_step.eval(), time.time()-start_time))
    sys.stdout.flush()
    model.global_epoch_step_op.eval()

  print('best test_gauc:', best_auc)
  sys.stdout.flush()
