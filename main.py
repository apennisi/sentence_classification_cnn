################################################################
#
#  Copyright 2017 Andrea Pennisi
#
#  This file  is distributed under the terms of the
#  GNU Lesser General Public License (Lesser GPL)
#
#
#  You can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  It is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  See <http://www.gnu.org/licenses/>.
#
#
#  It has been written by Andrea Pennisi
#
#
################################################################

import tensorflow as tf
import argparse
import sys
import os
import random
import numpy as np
from data import DataManager
from network import TextClassification

filters = [3,4,5]
emb_size = 128
db = DataManager()
checkpoint = 100
FLAGS = None
classes = 2


def generate_indices(l, nr_batch):
    indices = range(0, l)
    for _ in range(20):
        random.shuffle(indices)
    return indices[0:nr_batch]


def main(_):
    if not os.path.exists("models"):
        os.makedirs("models")

    max_seq, voc_size = db.load_data(FLAGS.first_cat, FLAGS.second_cat)
    net = TextClassification(voc_size, max_seq, emb_size)

    x = tf.placeholder(tf.int32, [None, max_seq], name="data")
    labels = tf.placeholder(tf.float32, [None, classes], name="labels")
    keep_prob = tf.placeholder(tf.float32)
    phase_train = tf.placeholder(tf.bool)

    scores = net.network(x, keep_prob, filters, phase_train)
    loss = net.loss(scores, labels)
    accuracy = net.accuracy(scores, labels)

    optimizer = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        len_val = len(db.test_labels)
        train_acc, val_acc, test_acc, cl_1_acc, cl_2_acc = [], [], [], [], []
        for i in range(FLAGS.epochs):
            data, l = db.get_batch_(FLAGS.n_batches, "Train")
            _, acc, ls = sess.run([optimizer, accuracy, loss], feed_dict={x: data, labels: l, keep_prob: 0.5, phase_train: True})
            print("Epoch %d/%d: loss %f acc %f" % (i, FLAGS.epochs, ls, acc))
            train_acc.append(acc)
            if i != 0 and i % checkpoint == 0:
                indices = generate_indices(len_val, FLAGS.n_batches)
                data, l = db.test_data[indices], db.test_labels[indices]
                acc = sess.run(accuracy, feed_dict={x: data, labels: l, keep_prob: 1.0, phase_train: False})
                val_acc.append(acc)
                print("Validation accuracy at epoch %d: %f" % (i, acc))
                path = saver.save(sess, "models/model", global_step=i)
                print("Model saved to {}\n".format(path))

        n_test = db.test_labels.shape[0]

        for i in range(n_test):
            #data, l = db.get_batch_(FLAGS.n_batches, "Test")
            data, l = db.test_data[i], db.test_labels[i]
            acc = sess.run(accuracy, feed_dict={x: np.expand_dims(data, 0), labels: np.expand_dims(l, 0),
                                                keep_prob: 1.0, phase_train: False})
            if l[1] == 1:
                cl_1_acc.append(acc)
            else:
                cl_2_acc.append(acc)
            test_acc.append(acc)
            print("Test %d/%d: acc %f" % (i, n_test, acc))

    tr_acc = np.nanmean(train_acc)
    tr_std = np.nanstd(train_acc)

    v_acc = np.nanmean(val_acc)
    v_std = np.nanstd(val_acc)

    te_acc = np.nanmean(test_acc)
    te_std = np.nanstd(test_acc)

    cl1_acc = np.nanmean(cl_1_acc)
    cl1_std = np.nanstd(cl_1_acc)

    cl2_acc = np.nanmean(cl_2_acc)
    cl2_std = np.nanstd(cl_2_acc)

    print("Training accuracy: %f std: %f" % (tr_acc, tr_std))
    print("Validation accuracy: %f std: %f" % (v_acc, v_std))
    print("Testing accuracy: %f std: %f" % (te_acc, te_std))
    print("Testing accuracy CL 1: %f std: %f samples %d" % (cl1_acc, cl1_std, len(cl_1_acc)))
    print("Testing accuracy CL 2: %f std: %f samples %d" % (cl2_acc, cl2_std, len(cl_2_acc)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_cat', type=str, required=True,
                        help='File containing the first category')
    parser.add_argument('--second_cat', type=str, required=True,
                        help='File containing the second category')
    parser.add_argument('--epochs', type=int, required=True,
                        help='Number of epochs')
    parser.add_argument('--n_batches', type=int, required=True,
                        help='Number of batches')
    parser.add_argument('--lr', type=float, required=True,
                        help='Learning Rate')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

