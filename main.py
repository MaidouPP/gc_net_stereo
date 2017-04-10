import tensorflow as tf
import h5py

def image_input():
  pass

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