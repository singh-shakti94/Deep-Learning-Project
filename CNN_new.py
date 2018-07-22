from __future__ import division, print_function, absolute_import
import tensorflow as tf
import pandas as pd
import numpy as np

# Import the data
data = pd.read_json("train.json")

# training data (for now using only band_1 for convolution)
# labels are in "is_iceberg" column where 0 value indicates a ship while 1 indicates iceberg
X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_1"]])
X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_2"]])
channel_3 = X_band_1 + X_band_2
train_data = np.concatenate([X_band_1[:, :, :, np.newaxis],
                             X_band_2[:, :, :, np.newaxis],
                             channel_3[:, :, :, np.newaxis]], axis=-1)
split = np.array_split(train_data, 10, axis=0)
train_data = np.concatenate(split[0:9], axis=0)
test_data = split[9]
# converting labels to one-hot vector
train_targets = data["is_iceberg"]
b = np.zeros((1604, 2))
b[np.arange(1604), list(train_targets)] = 1
train_targets = np.concatenate(np.array_split(b, 10, axis=0)[0:9], axis=0)
test_targets = np.array_split(b, 10, axis=0)[9]

# Training Parameters
learning_rate = 0.001
batch_size = 10

# Network Parameters
num_input = 5625 * 3  # flattened data input (input shape: 75*75*3)
num_classes = 2  # number of classes (is ship or iceberg)
dropout = 0.75  # Dropout, probability to keep units

# compute graph input


class CNN:
    def __init__(self):
        """
        In this class we will be performing 2-D convolution and training of the model.
        for now we are using 3 convolution layers and each layer is followed by a
        pooling operation.
        for each convolution layer, we are using a 5x5 filter with stride 1 on every dimension.
        ReLU activation is used after each convolution layer operation.
        after convolution layers, a fully connected layer of 1024 units is used to compute logits
        """
        self.X = tf.placeholder(tf.float32, [None, num_input])  # flattened input
        self.Y = tf.placeholder(tf.float32, [None, num_classes])  # one-hot vector of labels
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        self.weights = {
            # 3x3 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
            # 5x5 conv, 64 inputs, 128 outputs
            'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
            # fully connected, 8*8*128 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([10 * 10 * 256, 512])),
            # 1024 inputs, 2 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([512, num_classes]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([64])),
            'bc2': tf.Variable(tf.random_normal([128])),
            'bc3': tf.Variable(tf.random_normal([256])),
            'bd1': tf.Variable(tf.random_normal([512])),
            'out': tf.Variable(tf.random_normal([num_classes]))
        }

        # Construct compute graph
        logits = self.conv_net(self.X, self.weights, self.biases, self.keep_prob)
        prediction = tf.nn.softmax(logits)

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.loss_op)

        # model evaluation
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and ReLU activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, ksize=2, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, k, k, 1],
                              padding='SAME')

    # model creation
    def conv_net(self, x, weights, biases, dropout):
        # data input is a 1-D vector of 5625 features (75*75 data points)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 75, 75, 3])

        # Convolution Layer 1
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling
        conv1 = self.maxpool2d(conv1, k=2)

        # Convolution Layer 2
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling
        conv2 = self.maxpool2d(conv2, k=2)

        # Convolution Layer 3
        conv3 = self.conv2d(conv2, weights['wc3'], biases['bc3'])
        # Max Pooling
        conv3 = self.maxpool2d(conv3, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    def test_accuracy(self, data, targets):
        return self.session.run(self.accuracy, feed_dict={self.X: data,
                                                          self.Y: targets,
                                                          self.keep_prob:1.0})

    def train(self, train_data, train_targets, batch_size=1, epochs=1):
        sess = self.session

        print("training started ...")
        batch = 0
        test_datum = [test_data.flatten().transpose() for test_data in test_data]
        for epoch in range(epochs):
            for datum, target in zip([train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)],
                                     [train_targets[i:i + batch_size] for i in
                                      range(0, len(train_targets), batch_size)]):
                datum = np.array(datum)
                datum = [datum.flatten().transpose() for datum in datum]
                sess.run(self.train_op, feed_dict={self.X: datum,
                                                   self.Y: target,
                                                   self.keep_prob: dropout})
                batch += 1
                if batch % 10 == 0 or batch == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([self.loss_op, self.accuracy], feed_dict={self.X: datum,
                                                                         self.Y: target,
                                                                         self.keep_prob: 1.0})
                    print("Step " + str(batch) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))
                    print("\t Test accuracy : {}".format(self.test_accuracy(test_datum, test_targets)))

        print("\n\noptimization complete!")
        print("\t test accuracy : {}".format(self.test_accuracy(test_datum, test_targets)))

c = CNN()
c.train(train_data, train_targets, batch_size=2, epochs=2)
