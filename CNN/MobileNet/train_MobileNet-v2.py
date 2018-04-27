#-*- coding: utf-8 -*-
# 训练 MobileNet-v2
"""

"""
import tensorflow as tf
from mobilenet_v2 import mobilenetv2
from config import *
from utils import *

import time
import glob
import os


def load(sess, saver, checkpoint_dir):
    import re
    print("[*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print("[*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print("[*] Failed to find a checkpoint")
        return False, 0


def main():
    height=args.height
    width=args.width

    sess=tf.Session()

    # read queue
    glob_pattern = os.path.join(args.dataset_dir, '*.tfrecord')
    tfrecords_list = glob.glob(glob_pattern)
    filename_queue = tf.train.string_input_producer(tfrecords_list, num_epochs=None)
    img_batch, label_batch = get_batch(filename_queue, args.batch_size)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    inputs = tf.placeholder(tf.float32, [None, height, width, 3], name='input')

    logits, pred=mobilenetv2(inputs, num_classes=args.num_classes, is_train=args.is_train)

    # loss
    loss_ = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch, logits=logits))
    # L2 regularization
    l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    total_loss = loss_ + l2_loss

    # evaluate model, for classification
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.cast(label_batch, tf.int64))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # learning rate decay
    base_lr = tf.constant(args.learning_rate)
    lr_decay_step = args.num_samples // args.batch_size * 2  # every epoch
    global_step = tf.placeholder(dtype=tf.float32, shape=())
    lr = tf.train.exponential_decay(base_lr, global_step=global_step, decay_steps=lr_decay_step,
                                    decay_rate=args.lr_decay)
    # optimizer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.9, momentum=0.9)
        train_op = tf.train.AdamOptimizer(
            learning_rate=lr, beta1=args.beta1).minimize(total_loss)

    # summary
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('accuracy', acc)
    tf.summary.scalar('learning_rate', lr)
    summary_op = tf.summary.merge_all()

    # summary writer
    writer = tf.summary.FileWriter(args.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())

    # saver for save/restore model
    saver = tf.train.Saver()
    # load pretrained model
    step=0
    if not args.renew:
        print('[*] Try to load trained model...')
        could_load, step = load(sess, saver, args.checkpoint_dir)

    max_steps = int(args.num_samples / args.batch_size * args.epoch)

    print('START TRAINING...')
    for _step in range(step+1, max_steps+1):
        start_time=time.time()
        feed_dict = {global_step:_step, inputs:img_batch}
        # train
        _, _lr = sess.run([train_op, lr], feed_dict=feed_dict)
        # print logs and write summary
        if _step % 10 == 0:
            _summ, _loss, _acc = sess.run([summary_op, total_loss, acc],
                                       feed_dict=feed_dict)
            writer.add_summary(_summ, _step)
            print('global_step:{0}, time:{1:.3f}, lr:{2:.8f}, acc:{3:.6f}, loss:{4:.6f}'.format
                  (_step, time.time() - start_time, _lr, _acc, _loss))

        # save model
        if _step % 10 == 0:
            save_path = saver.save(sess, os.path.join(args.checkpoint_dir, args.model_name), global_step=_step)
            print('Current model saved in ' + save_path)

    tf.train.write_graph(sess.graph_def, args.checkpoint_dir, args.model_name + '.pb')
    save_path = saver.save(sess, os.path.join(args.checkpoint_dir, args.model_name), global_step=max_steps)
    print('Final model saved in ' + save_path)
    sess.close()
    print('FINISHED TRAINING.')


if __name__=='__main__':
    main()
