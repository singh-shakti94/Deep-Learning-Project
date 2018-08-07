from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

cols = pd.read_csv('inputForLogistic', nrows=1).columns

traindata=pd.read_csv('inputForLogistic',usecols=cols[:-1])
label=pd.read_csv('inputForLogistic',usecols=['is_iceberg'])



class Data_ANN:
    def __init__(self, hidden_units, activations):
        """
        Initialise the weights and build the compute graph. Use AdamOptimizer with default parameters.
        :param hidden_units - list of number of hidden units.
               Eg: [10,20] => Layer 1 has 10 hidden units and Layer 2 has 20.
        :param activations - list of activations for each of the hidden layers.
               Eg: [tf.nn.sigmoid, tf.nn.tanh]
        :param intializer - the reference to the function used for intializing the weights
        """
        # Define the placeholders
        self.input =tf.placeholder(dtype=tf.float32,shape=[None,8])


        self.expected_output = tf.placeholder(dtype=tf.int32,shape=[None,1])

        # Initialise the weights and biases. Use zeros for the biases.
        weights = [tf.Variable(dtype=tf.float32,initial_value=0.01*np.random.randn(8,hidden_units[0]))]
        biases = [tf.Variable(dtype=tf.float32,initial_value=np.zeros(shape=[1,hidden_units[0]]))]

        # Loop here.
        for i in range(len(hidden_units)-1):
            biases.append(tf.Variable(dtype=tf.float32,initial_value=np.zeros(shape=[1,hidden_units[i+1]])))
            weights.append(tf.Variable(dtype=tf.float32,initial_value=0.01*np.random.randn(hidden_units[i],hidden_units[i+1])))

        biases.append(tf.Variable(dtype=tf.float32, initial_value=np.zeros(shape=[1,2])))
        weights.append(tf.Variable(dtype=tf.float32, initial_value=0.01*np.random.randn(hidden_units[len(hidden_units)-1], 2)))



        def graph_Builder(x):
            for i in range(len(activations)):
                x = activations[i](tf.matmul(x, weights[i]) + biases[i])
            logit= tf.matmul(x, weights[len(activations)]) + biases[len(activations)]
            pred=tf.nn.softmax(logit)
            return logit,pred

        # Build the graph for computing output.
        self.output,self.pred=graph_Builder(self.input)

        # # Define the loss and accuracy here. (Refer Tutorial)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.expected_output, 2), logits=self.output))

        correct_prediction = tf.equal(tf.reshape(tf.argmax(self.output, 1), [-1, 1]),
                                      tf.cast(self.expected_output, dtype=tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #
        # # Instantiate the optimizer
        optimizer=tf.train.AdamOptimizer()
        #
        self.train_op = optimizer.minimize(self.cost)
        self.session = tf.Session()
        #
        # # Initialize all variables
        self.init=tf.global_variables_initializer()

    def train(self, train_data, train_labels, eval_data, eval_labels, batch_size, epochs=100):
        """
        Training code.
        """
        sess = self.session
        sess.run(self.init)

        # Slice the data and labels into batches depending on the batch_size.
        batches = [[train_data[k:k+batch_size],train_labels[k:k+batch_size]] for k in range(0,len(train_data),batch_size)]

        for epoch in range(epochs):
            cost_epoch = 0
            for batch in batches:
                # Forward Propagate, compute cost and backpropagate.
                cost, _ = sess.run([self.cost, self.train_op], feed_dict={self.input: batch[0],
                                                                          self.expected_output: batch[1]})
                cost_epoch += cost
            if epoch % 10 == 0:
                print("Train accuracy: {}".format(self.compute_accuracy(train_data, train_labels)))
                print("Test accuracy: {}".format(self.compute_accuracy(eval_data, eval_labels)))
            print("Epoch {}: {}".format(epoch, cost_epoch))
        print("Train accuracy: {}".format(self.compute_accuracy(train_data, train_labels)))
        print("Test accuracy: {}".format(self.compute_accuracy(eval_data, eval_labels)))

    def compute_accuracy(self, data, labels):
        """
        Fill in code to compute accuracy
        """

        _,acc= self.session.run([self.output,self.accuracy],feed_dict={self.input: data, self.expected_output: labels})

        return acc

ann = Data_ANN([1],[tf.nn.relu])
train_data, eval_data ,train_labels, eval_labels = train_test_split(traindata, label,
                                                        test_size=0.3)
ann.train(train_data,train_labels,eval_data,eval_labels,batch_size=1,epochs=10)