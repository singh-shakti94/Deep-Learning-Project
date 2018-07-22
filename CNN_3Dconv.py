from __future__ import division, print_function, absolute_import
import tensorflow as tf
import pandas as pd
import numpy as np

# Import the data
data = pd.read_json("train.json")

# training data (for now using only band_1 for convolution)
# labels are in "is_iceberg" column where 0 value indicates a ship while 1 indicates iceberg
# train_data = np.array(data[["band_1", "band_2", "is_iceberg"]])
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_2"]])
train_data = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis]], axis=-1)

train_targets = data["is_iceberg"]
# train_data.shape
# converting labels to one-hot vector
b = np.zeros((1604, 2))
b[np.arange(1604), list(train_targets)] = 1
train_targets = b

# Training Parameters
learning_rate = 0.01
batch_size = 10

# Network Parameters
num_input = 5625*2  # flattened band_1 data input (image shape: 75*75*2)
num_classes = 2  # number of classes (is ship or iceberg)
dropout = 0.75  # Dropout, probability to keep units

# compute graph input
X = tf.placeholder(tf.float32, [None, num_input])  # flattened input
Y = tf.placeholder(tf.float32, [None, num_classes])  # one-hot vector of labels
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


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
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([2, 5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([32, 5, 5, 32, 64])),
            # 5x5 conv, 64 inputs, 128 outputs
            'wc3': tf.Variable(tf.random_normal([64, 5, 5, 64, 128])),
            # fully connected, 10*10*128 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([10 * 10 * 128, 1024])),
            # 1024 inputs, 2 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, num_classes]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bc3': tf.Variable(tf.random_normal([128])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([num_classes]))
        }

        # Construct compute graph
        logits = self.conv_net(X, self.weights, self.biases, keep_prob)
        prediction = tf.nn.softmax(logits)

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.loss_op)

        # model evaluation
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def conv3d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and ReLU activation
        x = tf.nn.conv3d(x, W, strides=[1, 1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool3d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool3d(x, ksize=[1, k, k, k, 1], strides=[1, k, k, k, 1],
                              padding='SAME')

    # model creation
    def conv_net(self, x, weights, biases, dropout):
        # data input is a 1-D vector of 5625 features (75*75 data points)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 75, 75, 2, 1])

        # Convolution Layer 1
        conv1 = self.conv3d(x, weights['wc1'], biases['bc1'])
        # Max Pooling
        conv1 = self.maxpool3d(conv1, k=2)

        # Convolution Layer 2
        conv2 = self.conv3d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling
        conv2 = self.maxpool3d(conv2, k=2)

        # Convolution Layer 3
        conv3 = self.conv3d(conv2, weights['wc3'], biases['bc3'])
        # Max Pooling
        conv3 = self.maxpool3d(conv3, k=2)

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

    def train(self, train_data, train_targets, batch_size=1, epochs=1):
        sess = self.session

        print("training started ...")
        batch = 0
        for epoch in range(epochs):
            for datum, target in zip([train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)],
                                     [train_targets[i:i + batch_size] for i in
                                      range(0, len(train_targets), batch_size)]):
                datum = np.array(datum)
                print(datum.shape)
                print(target.shape)
                datum = [datum.flatten().transpose() for datum in datum]
                sess.run(self.train_op, feed_dict={X: datum, Y: target, keep_prob: dropout})
                batch += 1
                if batch % 10 == 0 or batch == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([self.loss_op, self.accuracy], feed_dict={X: datum,
                                                                         Y: target,
                                                                         keep_prob: 1.0})
                    print("Step " + str(batch) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))


        # for epoch in range(epochs):
        #     print("\nepoch : ", epoch)
        #     for step in range(len(batches_x)-1):
        #         batch_x = batches_x[step]
        #         batch_x = np.array([np.array(xi) for xi in batch_x])
        #         batch_y = batches_y[step]
        #         batch_y = np.array([np.array(yi) for yi in batch_y])
        #         # Run optimization op (Backpropagation)
        #         sess.run(self.train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
        #         if step % 10 == 0 or step == 1:
        #             # Calculate batch loss and accuracy
        #             loss, acc = sess.run([self.loss_op, self.accuracy], feed_dict={X: batch_x,
        #                                                                  Y: batch_y,
        #                                                                  keep_prob: 1.0})
        #             print("Step " + str(step) + ", Minibatch Loss= " + \
        #                   "{:.4f}".format(loss) + ", Training Accuracy= " + \
        #                   "{:.3f}".format(acc))

        print("Optimization Finished!")


c = CNN()
c.train(train_data, train_targets, batch_size=10, epochs=2)
