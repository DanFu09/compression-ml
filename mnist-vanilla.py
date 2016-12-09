import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import jpeg
import numpy as np
import time

sess = tf.InteractiveSession()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

## Model

# D = Input dimension, K = output dimension (# of classification categories)
D, K = 196, 10

K1 = 1024

X = tf.placeholder(tf.float32, [None, D])
Y_ = tf.placeholder(tf.float32, [None, K])

W0 = weight_variable([D, K1])
b0 = bias_variable([K1])

h_fc1 = tf.matmul(X, W0) + b0

W1 = weight_variable([K1, K])
b1 = bias_variable([K])

# Dropout to prevent overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

Y = tf.nn.softmax(tf.matmul(h_fc1_dropout, W1) + b1)

# We minimize the cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y), reduction_indices=[1]))
opt = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# And use the accuracy as a measure of success
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

# Train
def sigmoid(t):
    return 1. / (1. + np.exp(-t))

def minmax_scale(X):
    return X / (np.max(X) - np.min(X))

def mutate(X):
    return X

dct_truncate_D = lambda x: jpeg.dct_truncate(x, D, 28, 28)
def mutate_data(X):
    # Input comes in as a normalized bitmap between 0 and 1.
    # The below function just multiplies everything by 256
    X = np.apply_along_axis(jpeg.normal_to_bitmap, 1, X)

    # Actually applies the DCT and truncates the vector
    X = np.apply_along_axis(dct_truncate_D, 1, X)

    # We have to squash the input between 0 and 1 to make the network converge
    X = np.apply_along_axis(minmax_scale, 1, X)
    return X

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
X_test = mutate_data(mnist.test.images)
X_train = mutate_data(mnist.train.images)
Y_test = mnist.test.labels
Y_train = mnist.train.labels

j = 0
batch_size = 50
def get_next_batch(X, Y):
    global j
    if (j + 1) * batch_size > X.shape[0]:
        j = 0
    make_batch = lambda x: x[j * batch_size : (j + 1) * batch_size, :]

    j += 1
    return make_batch(X), make_batch(Y)

start = time.time()
for i in xrange(20000):
    batch = get_next_batch(X_train, Y_train)

    if i % 500 == 0:
        w0 = sess.run(W0, {X: X_test, Y_: Y_test, keep_prob: 1.0})
        y = Y.eval({X: X_test, Y_: Y_test, keep_prob: 1.0})
        print 'Train: ', accuracy.eval({X: batch[0], Y_: batch[1], keep_prob: 1.0})
        print 'Test {}: {}'.format(i, accuracy.eval({X: X_test, Y_: Y_test, keep_prob: 1.0}))
    opt.run(feed_dict={X: batch[0], Y_: batch[1], keep_prob: 0.5})
end = time.time()
print 'Time = {}'.format(end - start)
