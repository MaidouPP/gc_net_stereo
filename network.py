import tensorflow as tf
from config import Config
import numpy as np
import random

MOVING_AVERAGE_DECAY = 0.99
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00005
CONV_WEIGHT_STDDEV = 0.1
GC_VARIABLES = 'gc_variables'
UPDATE_OPS_COLLECTION = 'gc_update_ops'  # training ops
HEIGHT = 256
WIDTH = 512
DISPARITY = 192

# wrapper for 2d convolution op
def conv(x, c):
  ksize = c['ksize']
  stride = c['stride']
  filters_out = c['conv_filters_out']

  filters_in = x.get_shape()[-1]
  shape = [ksize, ksize, filters_in, filters_out]
  initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
  weights = _get_variable('weights',
                          shape=shape,
                          dtype='float',
                          initializer=initializer,
                          weight_decay=CONV_WEIGHT_DECAY)
  bias = tf.get_variable('bias', [filters_out], 'float', tf.constant_initializer(0.05, dtype='float'))
  x = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
  return tf.nn.bias_add(x, bias)

def conv_3d(x, c):
  ksize = c['ksize']
  stride = c['stride']
  filters_out = c['conv_filters_out']
  filters_in = x.get_shape()[-1]
  shape = [ksize, ksize, ksize, filters_in, filters_out]
  initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
  weights = _get_variable('weights',
                          shape=shape,
                          dtype='float',
                          initializer=initializer,
                          weight_decay=CONV_WEIGHT_DECAY)
  bias = tf.get_variable('bias', [filters_out], 'float', tf.constant_initializer(0.05, dtype='float'))
  x = tf.nn.conv3d(x, weights, [1, stride, stride, stride, 1], padding='SAME')
  return tf.nn.bias_add(x, bias)

def deconv_3d(x, c):
  ksize = c['ksize']
  stride = c['stride']
  filters_out = c['conv_filters_out']
  filters_in = x.get_shape()[-1]
  d = x.get_shape()[0] * stride
  height = x.get_shape()[1] * stride
  width = x.get_shape()[2] * stride
  output_shape = [d, height, weight, filters_out]
  strides = [1, stride, stride, stride, 1]
  shape = [ksize, ksize, ksize, filters_in, filters_out]
  initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
  weights = _get_variable('weights',
                          shape=shape,
                          dtype='float',
                          initializer=initializer,
                          weight_decay=CONV_WEIGHT_DECAY)
  bias = tf.get_variable('bias', [filters_out], 'float', tf.constant_initializer(0.05, dtype='float'))
  x = tf.nn.conv3d_transpose(x, weights, output_shape, strides, padding='SAME')
  return tf.nn.bias_add(x, bias)

# wrapper for batch-norm op
def bn(x, c):
  x_shape = x.get_shape()
  params_shape = x_shape[-1:]

  axis = list(range(len(x_shape) - 1))

  beta = _get_variable('beta',
                       params_shape,
                       initializer=tf.zeros_initializer)
  gamma = _get_variable('gamma',
                        params_shape,
                        initializer=tf.ones_initializer)

  moving_mean = _get_variable('moving_mean',
                              params_shape,
                              initializer=tf.zeros_initializer,
                              trainable=False)
  moving_variance = _get_variable('moving_variance',
                                  params_shape,
                                  initializer=tf.ones_initializer,
                                  trainable=False)

  # These ops will only be preformed when training.
  mean, variance = tf.nn.moments(x, axis)
  update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                             mean, BN_DECAY)
  update_moving_variance = moving_averages.assign_moving_average(
                                        moving_variance, variance, BN_DECAY)
  tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
  tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

  mean, variance = control_flow_ops.cond(
    c['is_training'], lambda: (mean, variance),
    lambda: (moving_mean, moving_variance))

  x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
  # x.set_shape(inputs.get_shape()) ??

  return x

# wrapper for get_variable op
def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
  "A little wrapper around tf.get_variable to do weight decay and add to"
  "resnet collection"
  if weight_decay > 0:
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    regularizer = None
  collections = [tf.GraphKeys.VARIABLES, GC_VARIABLES]
  return tf.get_variable(name,
                         shape=shape,
                         initializer=initializer,
                         dtype=dtype,
                         regularizer=regularizer,
                         collections=collections,
                         trainable=trainable)

# resnet block
def stack(x, c):
  shortcut = x
  with tf.variable_scope('block_A'):
    x = conv(x, c)
    x = bn(x, c)
    x = tf.nn.relu(x)
  with tf.variable_scope('block_B'):
    x = conv(x, c)
    x = bn(x, c)
    x = shortcut + x
    x = tf.nn.relu(x)

# siamese structure
def _build_resnet(x, c):
  imageL = tf.placeholder(tf.float32, shape=([1, HEIGHT, WIDTH]), name='L')
  imageR = tf.placeholder(tf.float32, shape=([1, HEIGHT, WIDTH]), name='R')

  with tf.variable_scope('downsample'):
    c['conv_filters_out'] = 32
    c['ksize'] = 5
    c['stride'] = 2
    x = conv(x, c)
    x = bn(x, c)
    x = tf.nn.relu(x)

  c['ksize'] = 3
  c['stride'] = 1

  with tf.variable_scope('resnet'):
    for i in xrange(c['num_resblock']):
      with tf.variable_scope('res' + str(i+1)):
        x = stack(x, c)

  return x

def _build_3d_conv(x, c):
  c['ksize'] = 3
  c['stride'] = 1
  c['conv_filters_out'] = 32
  for i in xrange(c['num_3d']):
    with tf.variable_scope(str(i)+'3d'):
      x = conv_3d(cost_vol, c)
      x = bn(x, c)
      x = tf.nn.relu(x)

      x = conv_3d(cost_vol, c)
      x = bn(x, c)
      x = tf.nn.relu(x)

      c['stride'] = 2
      if i is 3:
        c['conv_filters_out'] = 128
      else:
        c['conv_filters_out'] = 64
      x = conv_3d(cost_vol, c)
      x = bn(x, c)
      x = tf.nn.relu(x)

      c['stride'] = 1
      c['conv_filters_out'] = 64

  return x

def inference(left_x, right_x, is_training):
  # imageL = tf.placeholder(tf.float32, shape=([1, HEIGHT, WIDTH]), name='L')
  # imageR = tf.placeholder(tf.float32, shape=([1, HEIGHT, WIDTH]), name='R')
  c = Config()
  c['is_training'] = tf.convert_to_tensor(is_training,
                                          dtype = 'bool',
                                          name = 'is_training')
  c['num_resblock'] = 8  # totally 8 resnet blocks
  c['num_3d'] = 4  # totally 4 blocks of 3d conv

  with tf.variable_scope("siamese") as scope:
    left_features = _build_resnet(left_x, c)
    scope.reuse_variables()
    right_features = _build_resnet(right_x, c)

  # create cost volume
  cost_vol = tf.zeros([DISPARITY/2, HEIGHT/2, WIDTH/2, 2*c['conv_filters_out']], tf.float32)
  for d in xrange(DISPARITY/2):
    for k in xrange(c['conv_filters_out']):
      left_feature = tf.slice(left_features, [0, 0, k], [HEIGHT/2, WIDTH/2, 1])
      right_feature = tf.slice(right_features, [0, 0, k], [HEIGHT/2, WIDTH/2, 1])
      _, right_features_d = tf.split(right_feature, [i, WIDTH-i], 1)
      paddings = [[0, 0], [i, 0], [0, 0]]
      tf.pad(right_features_d, paddings, "CONSTANT")
      cost_vol[d, :, :, k] = left_features
      cost_vol[d, :, :, k+1] = right_features_d

  # 3d convolution
  with tf.variable_scope("3dconv"):
    c['ksize'] = 3
    c['stride'] = 1
    c['conv_filters_out'] = 32
    with tf.variable_scope(str(0) + '3d'):
      x = conv_3d(cost_vol, c)
      x = bn(x, c)
      x = tf.nn.relu(x)

      x = conv_3d(x, c)
      x = bn(x, c)
      x20 = tf.nn.relu(x)

      c['stride'] = 2
      c['conv_filters_out'] = 64

      x = conv_3d(x20, c)
      x = bn(x, c)
      x = tf.nn.relu(x)

    with tf.variable_scope(str(1) + '3d')
      c['stride'] = 1

      x = conv_3d(x, c)
      x = bn(x, c)
      x = tf.nn.relu(x)

      x = conv_3d(x, c)
      x = bn(x, c)
      x23 = tf.nn.relu(x)

      c['stride'] = 2

      x = conv_3d(x23, c)
      x = bn(x, c)
      x = tf.nn.relu(x)

    with tf.variable_scope(str(2) + '3d')
      c['stride'] = 1

      x = conv_3d(x, c)
      x = bn(x, c)
      x = tf.nn.relu(x)

      x = conv_3d(x, c)
      x = bn(x, c)
      x26 = tf.nn.relu(x)

      c['stride'] = 2

      x = conv_3d(x26, c)
      x = bn(x, c)
      x = tf.nn.relu(x)

    with tf.variable_scope(str(3) + '3d')
      c['stride'] = 1

      x = conv_3d(x, c)
      x = bn(x, c)
      x = tf.nn.relu(x)

      x = conv_3d(x, c)
      x = bn(x, c)
      x29 = tf.nn.relu(x)

      c['stride'] = 2
      c['conv_filters_out'] = 128

      x = conv_3d(x29, c)
      x = bn(x, c)
      x = tf.nn.relu(x)

      c['stride'] = 1
      x = conv_3d(x, c)
      x = bn(x, c)
      x = tf.nn.relu(x)

      x = conv_3d(x, c)
      x = bn(x, c)
      x = tf.nn.relu(x)


  # 3d deconvolution
  with tf.variable_scope("deconv"):
    c['stride'] = 2
    c['conv_filters_out'] = 64
    x = deconv_3d(x, c)
    x = bn(x, c)
    x = tf.nn.relu(x)
    x = x + x29

    x = deconv_3d(x, c)
    x = bn(x, c)
    x = tf.nn.relu(x)
    x = x + x26

    x = deconv_3d(x, c)
    x = bn(x, c)
    x = tf.nn.relu(x)
    x = x + x23

    c['conv_filters_out'] = 32
    x = deconv_3d(x, c)
    x = bn(x, c)
    x = tf.nn.relu(x)
    x = x + x20

    c['conv_filters_out'] = 1
    x = deconv_3d(x, c)

  # soft argmin 暂时先用分类的loss，而不是文章中的soft argmin的做法
  x = tf.squeeze(x)
  x = -x
  # with tf.name_scope('softmax'):
  #   max_axis = tf.reduce_max(x, 2, keep_dims=True)
  #   x = tf.exp(x-max_axis)
  #   normalize = tf.reduce_sum(x, 2, keep_dims=True)
  #   x = x/normalize

  return x