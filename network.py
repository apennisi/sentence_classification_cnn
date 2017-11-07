import tensorflow as tf

class TextClassification:
    @staticmethod
    def __normalize(x, n_out, phase_train):
        with tf.variable_scope('bn'):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def __conv(self, x, name, shape,phase_train):
        W = self.__weight_variable(shape, name)
        b = self.__bias_variable([shape[3]], name)
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
        conv = self.__normalize(conv, shape[3], phase_train)
        relu = tf.nn.relu(tf.nn.bias_add(conv, b))
        return self.__max_pool(relu, shape[0])

    def __fc(self, x, name, shape):
        b = self.__weight_variable([shape], name)
        with tf.variable_scope(name) as scope:
            fc = tf.contrib.layers.fully_connected(x, shape, activation_fn=None,
                                                     biases_initializer=None, scope=scope)
        return tf.nn.relu(tf.nn.bias_add(fc, b))

    def __max_pool(self, x, f_size):
        return tf.nn.max_pool(x, ksize=[1, self.__row_size - f_size + 1, 1, 1],
                              strides=[1, 1, 1, 1], padding='VALID')

    @staticmethod
    def __weight_variable(shape, name):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name+"W")

    @staticmethod
    def __bias_variable(shape, name):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name+"_bias")

    def network(self, data, keep_prob, filters, phase_train):

        #To be sure that this is executed on the cpu
        with tf.device('/cpu:0'):
            random_w = tf.Variable(tf.random_uniform([self.__voc_size, self.__emb_size], -1.0, 1.0), name="W")
            embeddings = tf.nn.embedding_lookup(random_w, data)
            embeddings = tf.expand_dims(embeddings, -1)

        conv_1 = self.__conv(embeddings, "conv1", [filters[0], self.__emb_size, 1, 128], phase_train)

        conv_2 = self.__conv(embeddings, "conv2", [filters[1], self.__emb_size, 1, 128], phase_train)

        conv_3 = self.__conv(embeddings, "conv3", [filters[2], self.__emb_size, 1, 128], phase_train)

        conv = tf.concat([conv_1, conv_2, conv_3], axis=3)

        conv_flat = tf.contrib.layers.flatten(conv)

        fc1 = self.__fc(conv_flat, "fc1", 128)

        fc1_drop = tf.nn.dropout(fc1, keep_prob)

        fc2 = self.__fc(fc1_drop, "fc2", 2)

        return fc2

    def loss(self, scores, labels):
        with tf.name_scope("loss"):
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                               if 'bias' not in v.name]) * 0.001
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=labels)
            return tf.reduce_mean(loss) + lossL2

    def accuracy(self, scores, labels):
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(scores, 1), tf.argmax(labels, 1))
            return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def __init__(self, vocabulary_size, row_size, emb_size):
        self.__voc_size = vocabulary_size
        self.__row_size = row_size
        self.__emb_size = emb_size