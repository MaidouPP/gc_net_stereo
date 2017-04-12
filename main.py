import tensorflow as tf
import h5py
import numpy as np
from network import *

tf.app.flags.DEFINE_string('train_dir', '/tmp/resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "learning rate.")
tf.app.flags.DEFINE_integer('max_steps', 1500000, "max steps")

def image_input():
  pass

def loss(logits, labels):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
  cross_entropy_mean = tf.reduce_mean(cross_entropy)

  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

  loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)

  return loss_

def train(logits, images, labels):
  global_step = tf.get_variable('global_step', [],
                                initializer=tf.constant_initializer(0),
                                trainable=False)

  loss_ = loss(logits, labels)
  # loss_avg
  ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
  # 对loss_保持移动平均
  tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
  # tf.scalar_summary('loss;', ema.average(loss_))
  tf.scalar_summary('loss: ', loss_)
  tf.scalar_summary('learning_rate', FLAGS.learning_rate)

  opt = tf.train.RMSPropOptimizer(FLAGS.learning_rate, 0.99)
  grads = opt.compute_gradients(loss_)
  # this step also adds 1 to global_step
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
  batchnorm_updates_op = tf.group(*batchnorm_updates)
  train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

  saver = tf.train.Saver(tf.all_variables())
  summary_op = tf.merge_all_summaries()

  init = tf.initialize_all_variables()

  sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
  sess.run(init)

  summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

  for x in xrange(FLAGS.max_steps + 1):
    start_time = time.time()

    step = sess.run(global_step)
    i = [train_op, loss_]

    write_summary = step % 100 and step > 1
    if write_summary:
      i.append(summary_op)

    o = sess.run(i, {is_training: True})

    loss_value = o[1]

    duration = time.time() - start_time

    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    if step % 5 == 0:
      examples_per_sec = 1 / float(duration)
      format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
      print(format_str % (step, loss_value, examples_per_sec, duration))

    if write_summary:
      summary_str = o[2]
      summary_writer.add_summary(summary_str, step)

    # Save the model checkpoint periodically.
    if step > 1 and step % 100 == 0:
      checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
      saver.save(sess, checkpoint_path, global_step=global_step)

    # Run validation periodically
    if step > 1 and step % 100 == 0:
      _, top1_error_value = sess.run([val_op, top1_error], {is_training: False})
      print('Validation top1 error %.2f' % top1_error_value)

def main(_):
  images, labels = image_input()

  logits = inference(images,
                     num_classes=1000,
                     is_training=True,
                     bottleneck=False,
                     num_blocks=[2, 2, 2, 2])
  train(logits, images, labels)


if __name__ == '__main__':
  tf.app.run()