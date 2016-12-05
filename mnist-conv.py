from sklearn import preprocessing
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import jpeg
import numpy as np
from sklearn import preprocessing

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()

## Model
# D = Input dimension, K = output dimension (# of classification categories)
D, K = 28 * 28, 10

X = tf.placeholder(tf.float32, [None, D])
Y_ = tf.placeholder(tf.float32, [None, K])

x_image = tf.reshape(X, [-1,28,28,1])

# First convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout to prevent overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Output layer
W_fc2 = weight_variable([1024, K])
b_fc2 = bias_variable([K])

Y=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# We minimize the cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y), reduction_indices=[1]))
opt = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# And use the accuracy as a measure of success
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

dct_truncate_D = lambda x: jpeg.dct_truncate(x, D)

# Train
def mutate(X):
    return X

def mutate_data(X):
    return X
    return X + 1
    X = np.apply_along_axis(jpeg.normal_to_bitmap, 1, X)
    X = np.apply_along_axis(dct_truncate_D, 1, X)
    return preprocessing.scale(X)

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

for i in xrange(1000):

    batch = get_next_batch(X_train, Y_train)

    if i % 50 == 0:
        print Y.eval({X: X_test, Y_: Y_test, keep_prob: 1.0})
        print accuracy.eval({X: X_test, Y_: Y_test, keep_prob: 1.0})
    opt.run(feed_dict={X: batch[0], Y_: batch[1], keep_prob: 0.5})
