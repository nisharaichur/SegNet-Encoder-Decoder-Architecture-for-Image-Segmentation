import json
import numpy as np
import math
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os
import scipy
from scipy import misc
import cv2
import glob
import numpy as np

def initialization(k, c):
    std = math.sqrt(2. / (k ** 2 * c))
    return tf.truncated_normal_initializer(stddev=std)

def conv_layer(x, name, shape,  vgg_weights=None, use_vgg=False):

    def init_weight(val_name, vgg_weights):
        return vgg_weights[val_name][0]

    def init_bias(val_name, vgg_weights):
        return vgg_weights[val_name][1]


    with tf.variable_scope(name) as scope:
        if use_vgg:
            print("here")
            conv_init = tf.constant_initializer(init_weight(scope.name, vgg_weights))
            conv_filt = variable_with_weight_decay("weights", initializer=conv_init, shape=shape, wd=False)
            #conv_filt = tf.get_variable('weights', shape=shape, initializer=conv_init)

            bias_init = tf.constant_initializer(init_bias(scope.name, vgg_weights))
            bias_filt = variable_with_weight_decay("biases", initializer=bias_init, shape=shape[3], wd=False)
            #bias_filt = tf.get_variable('biases', shape=shape[3], initializer=bias_init)
        else:
            conv_filt = tf.get_variable("weights", shape=shape, initializer=initialization(shape[0], shape[2]))
            bias_filt = tf.get_variable('biases', shape=shape[3], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, conv_filt, strides=[1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, bias_filt)
        conv_out = tf.nn.relu(batch_norm(bias, scope))
    return conv_out



def variable_with_weight_decay(name, initializer, shape, wd):
  var = tf.get_variable(name, shape, initializer=initializer)
  weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
  tf.add_to_collection('losses', weight_decay)
  return var


def batch_norm(bias_input, scope):
  with tf.variable_scope(scope.name) as scope:
    return tf.contrib.layers.batch_norm(bias_input, is_training=True, center=False, scope=scope)

def max_pool(inputs, name):
    with tf.variable_scope(name) as scope:
        value, index = tf.nn.max_pool_with_argmax(tf.to_double(inputs), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope.name)

    return tf.to_float(value), index, inputs.get_shape().as_list()

def up_sampling(pool, ind, output_shape, batch_size, name=None):
    with tf.variable_scope(name):
        pool_ = tf.reshape(pool, [-1])
        batch_range = tf.reshape(tf.range(batch_size, dtype=ind.dtype), [tf.shape(pool)[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [-1, 1])
        ind_ = tf.reshape(ind, [-1, 1])
        ind_ = tf.concat([b, ind_], 1)
        ret = tf.scatter_nd(ind_, pool_, shape=[batch_size, output_shape[1] * output_shape[2] * output_shape[3]])
        ret = tf.reshape(ret, [tf.shape(pool)[0], output_shape[1], output_shape[2], output_shape[3]])
        return ret

def get_filename_list(path, config):
    fd = open(path)
    image_filenames = []
    label_filenames = []
    for i in fd:
        i = i.strip().split(" ")
        image_filenames.append(i[0])
        label_filenames.append(i[1])
    #print(image_filenames)
    image_filenames = ['/content/drive/My Drive/SegNet'+ name for name in image_filenames]
    label_filenames = ['/content/drive/My Drive/SegNet'+ name for name in label_filenames]
    return image_filenames, label_filenames

def generate_data(images, labels):
  images_num = []
  labels_num = []
  for x, y in zip(images, labels):
    image = cv2.imread(x)
    images_num.append(image)
    label = cv2.imread(y)
    labels_num.append(label)
  return np.array(images_num), np.array(labels_num)

def cal_loss(logits, labels, num_class=12):
  #class weights relative to number of such classes
  loss_weight = np.array([ 
                          0.2595,
                          0.1826,
                          4.5460,
                          0.1417,
                          0.9051,
                          0.3826,
                          9.6446,
                          1.8418,
                          0.6823,
                          6.2478,
                          7.3614,
                          1.0974
                          ])
  labels = tf.to_int64(labels)
  label_flatten = tf.reshape(labels, [-1])
  label_onehot = tf.one_hot(label_flatten , num_class)
  logit_reshape = tf.reshape(logits, [-1, num_class])
  cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=label_onehot, logits=logit_reshape, pos_weight=loss_weight)
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  correct_prediction = tf.equal(tf.argmax(logit_reshape, -1), label_flatten)
  accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
  return cross_entropy_mean, accuracy, tf.argmax(logit_reshape, -1)


#Credits: https://github.com/toimcio/SegNet-tensorflow
def per_class_acc(predictions, label_tensor, num_class):
    labels = label_tensor
    size = predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    for i in range(size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    acc_total = np.diag(hist).sum() / hist.sum()
    print('accuracy = %f' % np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('mean IU  = %f' % np.nanmean(iu))
    for ii in range(num_class):
        if float(hist.sum(1)[ii]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("class # %d accuracy = %f " % (ii, acc))


#Credits: https://github.com/toimcio/SegNet-tensorflow
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


#Credits: https://github.com/toimcio/SegNet-tensorflow
def get_hist(predictions, labels):
    num_class = predictions.shape[3]
    batch_size = predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    for i in range(batch_size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    return hist


#Credits: https://github.com/toimcio/SegNet-tensorflow
def print_hist_summary(hist):
    acc_total = np.diag(hist).sum() / hist.sum()
    print('accuracy = %f' % np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('mean IU  = %f' % np.nanmean(iu))
    for ii in range(hist.shape[0]):
        if float(hist.sum(1)[ii]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("class # %d accuracy = %f " % (ii, acc))

def train_op(total_loss):
  print("Training using SGD optimizer")
  global_step = tf.Variable(0, trainable=False)
  optimizer = tf.keras.optimizers.SGD(0.1, momentum=0.9)
  #optimizer = tf.train.AdamOptimizer(0.001, epsilon=0.0001)
  grads = optimizer.compute_gradients(total_loss, tf.trainable_variables())
  training_op = optimizer.apply_gradients(grads, global_step=global_step)
  return training_op, global_step, grads

