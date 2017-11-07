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
import numpy as np
import re
import random
import sys

class DataManager:

    @staticmethod
    def __clean_string(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    @staticmethod
    def __vocabulary(text):
        max_row_length = max([len(row.split(" ")) for row in text])
        voc_process = tf.contrib.learn.preprocessing.VocabularyProcessor(max_row_length)
        vocabulary = np.array(list(voc_process.fit_transform(text)))
        return vocabulary, max_row_length, len(voc_process.vocabulary_)

    @staticmethod
    def __batch(data, labels, idx, nr_batch):
        start = idx * nr_batch
        end = (idx + 1) * nr_batch
        if end > data.shape[0]:
            batch_data = data[start:]
            batch_labels = labels[start:]
            data_ok = False
        else:
            batch_data = data[start:end]
            batch_labels = labels[start:end]
            if end == data.shape[0]:
                data_ok = False
            else:
                data_ok = True
        return batch_data, batch_labels, data_ok

    def load_data(self, text_1, text_2):
        t_1 = list(open(text_1).readlines())
        t_2 = list(open(text_2).readlines())
        t_1 = [self.__clean_string(c.strip()) for c in t_1]
        t_2 = [self.__clean_string(c.strip()) for c in t_2]
        text = t_1 + t_2

        t_1_labels = [[0, 1] for _ in t_1]
        t_2_labels = [[1, 0] for _ in t_2]
        labels = np.concatenate([t_1_labels, t_2_labels], 0)

        vocabulary, seq_cat, voc_size = self.__vocabulary(text)

        #Mix the data
        indices = range(0, len(labels))
        random.shuffle(indices)
        random.shuffle(indices)
        labels = labels[indices]
        vocabulary = vocabulary[indices]

        #use the 30% of the data as testing
        idx = int(len(labels) - len(labels)*0.3)
        self.train_data, self.test_data = vocabulary[:idx], vocabulary[idx:]
        self.train_labels, self.test_labels = labels[:idx], labels[idx:]

        return seq_cat, voc_size

    def get_batch_(self, nr_batch, data = "Train"):
        if data is "Train":
            data_batch, label_batch, reset = \
                self.__batch(self.train_data, self.train_labels, self.__idx_train, nr_batch)
            self.__idx_train = 0 if reset is False else self.__idx_train + 1
        elif data is "Test":
            data_batch, label_batch, reset = \
                self.__batch(self.test_data, self.test_labels, self.__idx_test, nr_batch)
            self.__idx_test = 0 if reset is False else self.__idx_test + 1
        else:
            print("No known dataset!")
            sys.exit()
        return data_batch, label_batch

    def __init__(self):
        self.__idx_train = 0
        self.__idx_test = 0