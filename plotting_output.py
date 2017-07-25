
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)
encode_decode = pickle.load(open("./outputs/naive_en/init_data.pickle", 'rb'))

examples_to_show = 10

# plotting them
f, a = plt.subplots(2, 10, figsize = (10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
plt.savefig('./outputs/naive_en/version_test.png')
f.show()
plt.draw()

plt.waitforbuttonpress()
