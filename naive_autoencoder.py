"""
First example of autoencoders in tensorflow on MNIST dataset
Author: Akash Rastogi
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

# Hyper parameters
learning_rate = 0.005
training_epochs = 100
batch_size = 256
display_step = 1
examples_to_show = 10

# Architectural hyperparameters
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784

# Variable in the computational map
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases = {
    'encoder_h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_input]))
}

# Defining the entire encoder and decoder behaviour
def base_arch(inp_batch, weight, bias):
    output = tf.nn.sigmoid(tf.add(tf.matmul(inp_batch, weight), bias))
    return output

# Model construction
encoder_1 = base_arch(X, weights['encoder_h1'], biases['encoder_h1'])
encoder_2 = base_arch(encoder_1, weights['encoder_h2'], biases['encoder_h2'])
decoder_1 = base_arch(encoder_2, weights['decoder_h1'], biases['decoder_h1'])
decoder_2 = base_arch(decoder_1, weights['decoder_h2'], biases['decoder_h2'])

# Optimizer definition
cost = tf.reduce_mean(tf.pow(X - decoder_2, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Running the graph
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        if epoch % display_step == 0:
            print("Epoch: {0}, cost= {1:.2f}".format(epoch+1, c))

    print("Optimization done!")

    encode_decode = sess.run( decoder_2, feed_dict={X: mnist.test.images})

    fileObj = open("./outputs/naive_en/init_data.pickle", "wb")
    pickle.dump(encode_decode, fileObj)
    fileObj.close()



    #plotting them
    # f, a = plt.subplots(2, 10, figsize = (10, 2))
    # for i in range(examples_to_show):
    #     a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    #     a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    # plt.savefig('./outputs/naive_en/version_test.png')
    # f.show()
    # plt.draw()
    #
    # plt.waitforbuttonpress()
