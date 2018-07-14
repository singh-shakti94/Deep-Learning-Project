import numpy as np
import pandas as pd
import tensorflow as tf
data1 = pd.read_json("train.json")

width=75
height=75
depth=3
epochs=1
batch_size=1
nLabel=2

data2 = pd.read_json("train.json")
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data2["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data2["band_2"]])
train_data = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)
train_targets=np.array(data2["is_iceberg"]).reshape([-1,1])
print train_targets.shape
print train_data.shape
print train_data[0].shape

class CNN_3D:
  def __init__(self):

    self.input = tf.placeholder(tf.float32, shape=[None, width*height*depth]) # [None, 75*75*3]
    self.expected = tf.placeholder(tf.int32, shape=[None, 1])  # [None, 2]

    W_conv1 = self._weight_variable([1, 3, 3, 1, 32])
    b_conv1 = self._bias_variable([32])

    W_fc = self._weight_variable([92416, 1024])
    b_fc = self._bias_variable([1024])

    W_out = self._weight_variable([1024, nLabel])
    b_out = self._bias_variable([nLabel])


    x_image = tf.reshape(self.input, [-1, width, height, depth, 1])
    print(x_image.get_shape)


    h_conv1 = tf.nn.relu(self.conv3d(x_image, W_conv1) + b_conv1)
    print(h_conv1.get_shape)
    h_pool1 = self.max_pool_2x2(h_conv1)
    print(h_pool1.get_shape)
    fc = tf.reshape(h_pool1, [-1, 92416])
    fc = tf.nn.relu(tf.matmul(fc, W_fc) + b_fc)

    logits = tf.matmul(fc, W_out) + b_out
    self.output = tf.nn.softmax(logits, axis=1)
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.expected, 2), logits=logits)
    self.cost = tf.reduce_mean(entropy)

    correct_prediction = tf.equal(tf.reshape(tf.argmax(self.output, 1), [-1, 1]),
                                  tf.cast(self.expected, dtype=tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer()
    self.train_op = optimizer.minimize(self.cost)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())

  def _weight_variable(self,shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def _bias_variable(self,shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv3d(self,x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME') # conv2d, [1, 1, 1, 1]

  # Pooling: max pooling over 2x2 blocks
  def max_pool_2x2(self,x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

  def train(self, train_data,train_targets,batch_size=1, epochs=1):
    sess = self.session
    print("Starting training...")

    for epoch in range(epochs):

      cost_epoch = 0
      c = 0
      for datum, target in zip([train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)],
                               [train_targets[i:i + batch_size] for i in
                                range(0, len(train_targets), batch_size)]):
        datum=np.array(datum)
        datum=[datum.flatten().transpose() for datum in datum]
        _, cost = sess.run([self.train_op, self.cost],
                           feed_dict={self.input: datum, self.expected: target})

        cost_epoch += cost
        print cost_epoch

c=CNN_3D()
c.train(train_data,train_targets)
