#-*- coding: utf-8 -*-
# 预处理 
"""
"""
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize, imsave


def preprocess(image):
    # subtract mean
    mean=np.array([123.68, 116.779, 103.939])
    image=image-mean
    # scale to 1
    img = image * 0.017
    # return value should be float!
    return img

# tfrecord example features
def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# read tf_record
def read_tfrecord(filename_queue):
    feature = {'image/encoded': tf.FixedLenFeature([], tf.string),
               'image/height': tf.FixedLenFeature([], tf.int64),
               'image/width': tf.FixedLenFeature([], tf.int64),
               'image/label': tf.FixedLenFeature([], tf.int64)}

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features=feature)

    image  = tf.decode_raw(features['image/encoded'], tf.uint8)
    image  = tf.cast(image, tf.float32)
    height = tf.cast(features['image/height'],tf.int32)
    width  = tf.cast(features['image/width'], tf.int32)
    label  = tf.cast(features['image/label'], tf.int32)
    img = tf.reshape(image, [height, width, 3])

    # preprocess
    # subtract mean valu
    rgb_mean=np.array([123.68, 116.779, 103.939])
    img = tf.subtract(img, rgb_mean)
    # red, green, blue = tf.split(3, 3, img)
    # img = tf.concat(3, [
    #     tf.subtract(red , bgr_mean[2]),
    #     tf.subtract(green , bgr_mean[1]),
    #     tf.subtract(blue , bgr_mean[0]),
    # ])
    # center_crop
    img = tf.image.resize_images(img, [256, 256])
    j = int(round((256 - 224) / 2.))
    i = int(round((256 - 224) / 2.))
    img = img[j:j+224, i:i+224, :]

    # scale to 1
    img = tf.cast(img, tf.float32) * 0.017

    return img, label

def get_batch(infile, batch_size, num_threads=4, shuffle=False, min_after_dequeue=None):
    # 使用batch，img的shape必须是静态常量
    image, label = read_tfrecord(infile)

    if min_after_dequeue is None:
        min_after_dequeue = batch_size * 10
    capacity = min_after_dequeue + 3 * batch_size

    if shuffle:
        img_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                    capacity=capacity,num_threads=num_threads,
                                                    min_after_dequeue=min_after_dequeue)
    else:
        img_batch, label_batch = tf.train.batch([image, label], batch_size,
                                                capacity=capacity, num_threads=num_threads,
                                                allow_smaller_final_batch=True)

    return img_batch, label_batch
