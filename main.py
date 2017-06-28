import tensorflow as tf
import h5py
import numpy as np
from network import *
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import sys
import readPFM
from PIL import Image
import argparse

# FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('output_dir', '/home/users/shixin.li/segment/gc-net/log', "save log and checkpoints.")
# tf.app.flags.DEFINE_float('learning_rate', 0.001, "learning rate.")
# tf.app.flags.DEFINE_integer('max_steps', 50000, "max steps")
WIDTH=512
HEIGHT=256

def parse_args():
  parser = argparse.ArgumentParser(description="Tune stereo matching network")
  parser.add_argument('--phase', default='train',
                      help='train or test')
  parser.add_argument('--gpu', default='0',
                      help='state the index of gpu: 0, 1, 2 or 3')
  parser.add_argument('--output_dir', default='/home/users/shixin.li/segment/gc-net/log')
  parser.add_argument('--learning_rate', default=0.001, type=float)
  parser.add_argument('--max_steps', default=50000, type=int)
  parser.add_argument('--pretrain', default='false', 
                      help='true or false')
  args = parser.parse_args()
  return args

# predict the final disparity map
def predict(logits):
# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
# mask the invalid labels (>192)
# cross_entropy_valid = tf.where(mask, cross_entropy, tf.zeros_like(cross_entropy))
# cross_entropy_mean = tf.reduce_mean(cross_entropy_valid)
  softmax = tf.nn.softmax(logits)
  disp = tf.range(1, DISPARITY+1, 1)
  disp = tf.cast(disp, tf.float32)
  disp_mat = []
  for i in xrange(WIDTH*HEIGHT):
    disp_mat.append(disp)
  disp_mat = tf.reshape(tf.stack(disp_mat), [HEIGHT, WIDTH, DISPARITY])
  result = tf.multiply(softmax, disp_mat)
  result = tf.reduce_sum(result, 2)
  return result

def loss(logits, labels):
  mask = tf.cast(labels<=DISPARITY, dtype=tf.bool)
  loss_ = tf.abs(tf.subtract(logits, labels))
  loss_ = tf.where(mask, loss_, tf.zeros_like(loss_))
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  loss_sum = tf.reduce_sum(loss_)
  mask = tf.cast(mask, tf.float32)
  loss_mean = tf.div(loss_sum, tf.reduce_sum(mask))

  loss_final = tf.add_n([loss_mean] + regularization_losses)

  return loss_final

def normalizeRGB(img):
  img=img.astype(float)
  for i in range(3):
	  minval=img[:, :, i].min()
	  maxval=img[:, :, i].max()
	  if minval!=maxval:
		  img[:, :, i]=(img[:, :, i]-minval)/(maxval-minval)
  return img		

def train(dataset, args):
  tf.logging.set_verbosity(tf.logging.ERROR)
  with tf.device('/gpu:' + args.gpu):
    left_img = tf.placeholder(tf.float32, shape=(1, None, None, 3))
    right_img = tf.placeholder(tf.float32, shape=(1, None, None, 3))
    labels = tf.placeholder(tf.float32, shape=(None, None))

    logits = inference(left_img, right_img, is_training=True)
    logits = tf.transpose(logits, [1, 2, 0])
    
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    result = predict(logits)
    count_equal = tf.abs(tf.subtract(result, labels))
    # mark as positive if difference is smaller than 1px
    count_equal = tf.cast(count_equal<=1.0, dtype=tf.bool)
    correct_pixels = tf.reduce_sum(tf.to_int32(count_equal))
    correct_pixels = tf.cast(correct_pixels, tf.float32)
    pixels = tf.cast(tf.constant([WIDTH*HEIGHT]), tf.float32)
    # Here I forget to cast a mask to only compute accuracy based on valid pixels (<192)
    # Set it for now, better modify later!  --shixin
    accuracy = tf.div(correct_pixels, pixels)

    loss_ = loss(result, labels)

    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # keep moving average for loss
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    # tf.scalar_summary('loss;', ema.average(loss_))
    # tf.scalar_summary('loss: ', loss_)
    tf.summary.scalar('learning_rate', args.learning_rate)

    opt = tf.train.RMSPropOptimizer(args.learning_rate, decay=0.9, epsilon=1e-8)
    grads = opt.compute_gradients(loss_)
    # this step also adds 1 to global_step
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)

    # train op contains optimizer, batchnorm and averaged loss
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.initialize_all_variables()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # keep training from last checkpoint
    if args.pretrain == 'true':
      new_saver = tf.train.Saver()
      print "================= Loading checkpoint =============="
      new_saver.restore(sess, tf.train.latest_checkpoint('./log/0508'))
    else:
      sess.run(init)

    summary_writer = tf.summary.FileWriter(args.output_dir+'log', sess.graph)

    print "Start training.......................................!"
    for x in xrange(args.max_steps + 1):
      start_time = time.time()

#      with h5py.File(dataset, "r") as f:
#        randomNumber = str(random.randint(0, 3))
#        randomH = random.randint(0, image_height - HEIGHT - 1)
#        randomW = random.randint(0, image_width - WIDTH - 1)
#        imgL = f['left/' + randomNumber][()]
#        imgL = imgL[randomH:randomH + HEIGHT, randomW:randomW + WIDTH]
#        imgL = imgL.astype(float)
#        imgL = np.expand_dims(imgL, axis=0)
#        imgL = np.expand_dims(imgL, axis=3)
#
#        imgR = f['right/' + randomNumber][()]
#        imgR = imgR[randomH:randomH + HEIGHT, randomW:randomW + WIDTH]
#        imgR = imgR.astype(float)
#        imgR = np.expand_dims(imgR, axis=0)
#        imgR = np.expand_dims(imgR, axis=3)
#
#        imgGround = f['groundtruth/' + randomNumber][()]
#        imgGround = imgGround[randomH:randomH + HEIGHT, randomW:randomW + WIDTH]
#        imgGround = imgGround.astype(int)
      
      randomNumber = random.randint(0, 800-1)
      randomH = random.randint(0, image_height - HEIGHT - 1)
      randomW = random.randint(0, image_width - WIDTH -1)
      with open("train.lst", "r") as lst:
        img = lst.readlines()[randomNumber]
        img = img.strip()
      
      imgL = np.array(Image.open(dataset+"/left/"+img))
      imgL = normalizeRGB(imgL)
      imgL = imgL[randomH:randomH+HEIGHT, randomW:randomW+WIDTH, :]
      imgL = np.expand_dims(imgL, axis=0)
      
      imgR = np.array(Image.open(dataset+"/right/"+img))
      imgR = normalizeRGB(imgR)
      imgR = imgR[randomH:randomH+HEIGHT, randomW:randomW+WIDTH, :]
      imgR = np.expand_dims(imgR, axis=0)

      imgGround = readPFM.load_pfm(dataset+"/left_gt/"+img[:-4]+".pfm")
      imgGround = imgGround[randomH:randomH+HEIGHT, randomW:randomW+WIDTH]
      
      step = sess.run(global_step)
      i = [train_op, loss_, accuracy]


      write_summary = step % 100
      if write_summary or step==0:
        i.append(summary_op)

      o = sess.run(i, feed_dict={left_img: imgL, right_img: imgR, labels: imgGround})
      
      loss_value = o[1]
      accuracy_train = o[2]

      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        examples_per_sec = 1 / float(duration)
        format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch) accuracy = %.5f')
        print(format_str % (step, loss_value, examples_per_sec, duration, accuracy_train))

#      if write_summary:
#        summary_str = o[2]
#        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if (step > 1 and step % 200 == 0) or step==args.max_steps:
        checkpoint_path = os.path.join(args.output_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=global_step)

def test(dataset, args):
    # new_saver = tf.train.import_meta_graph("model.ckpt-141601.meta")
    
    # construct exactly same variables as training phase
    print "================ Building the same network as training phase ==============="
    left_img = tf.placeholder(tf.float32, shape=(1, HEIGHT, WIDTH, None))
    right_img = tf.placeholder(tf.float32, shape=(1, HEIGHT, WIDTH, None))
    labels = tf.placeholder(tf.float32, shape=(HEIGHT, WIDTH))

    logits = inference(left_img, right_img, is_training=True)
    logits = tf.transpose(logits, [1, 2, 0])
    
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    result = predict(logits)
    result_int = tf.cast(result, tf.int32)

    labels_int = tf.cast(labels, tf.int32)
    count_equal = tf.subtract(result_int, labels_int)
    count_equal = tf.equal(count_equal, tf.zeros_like(count_equal))
    correct_pixels = tf.reduce_sum(tf.to_int32(count_equal))
    correct_pixels = tf.cast(correct_pixels, tf.float32)
    pixels = tf.cast(tf.constant([WIDTH*HEIGHT]), tf.float32)
    accuracy = tf.div(correct_pixels, pixels)

    loss_ = loss(result, labels)
    
    sess = tf.Session()
    new_saver = tf.train.Saver()
    print "================= Loading checkpoint =============="
    new_saver.restore(sess, tf.train.latest_checkpoint(output_dir))

    with open("train.lst", "r") as lst:
      randomNumber = 0
      randomH = 0
      randomW = 0
      img = lst.readlines()[randomNumber]
      img = img.strip()
      
      imgL = np.array(Image.open(dataset+"/left/"+img))
      imgL = normalizeRGB(imgL)
      imgL = imgL[randomH:randomH+HEIGHT, randomW:randomW+WIDTH, :]
      imgL = np.expand_dims(imgL, axis=0)
     
      imgR = np.array(Image.open(dataset+"/right/"+img))
      imgR = normalizeRGB(imgR)
      imgR = imgR[randomH:randomH+HEIGHT, randomW:randomW+WIDTH, :]
      imgR = np.expand_dims(imgR, axis=0)

      imgGround = readPFM.load_pfm(dataset+"/left_gt/"+img[:-4]+".pfm")
      imgGround = imgGround[randomH:randomH+HEIGHT, randomW:randomW+WIDTH]
      print imgGround
    
    print "Start testing for one image.................."  
    o = sess.run([result, loss_], feed_dict={left_img: imgL, right_img: imgR, labels: imgGround})
    test_img = o[0].squeeze()
    test_img = np.asarray(test_img, np.uint8)
    test_img = Image.fromarray(test_img)
    print "result is: ", o[0]
    print "loss is: ", o[1]
    print "Saving test.png..................."
    test_img.save("test.png")
 

def main(_):
  global image_height
  global image_width
  image_height = 540
  image_width = 960
  global HEIGHT
  global WIDTH
  HEIGHT = 256
  WIDTH = 512
  dataset = "/home/users/shixin.li/segment/data_stereo"
  args = parse_args()
  if args.phase == 'train':
    train(dataset, args)
  else:
    HEIGHT=512
    WIDTH=960
    test(dataset, args)

if __name__ == '__main__':
  tf.app.run()
