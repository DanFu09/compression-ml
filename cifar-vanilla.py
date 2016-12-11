import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import jpeg
import numpy as np
from sklearn.decomposition import PCA
from ml_util import ml
import pdb

import time
# import cv2
# from matplotlib import pyplot as plt

sess = tf.InteractiveSession()

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

## Model

# D = Input dimension, K = output dimension (# of classification categories)
D, K = 192, 10

K1 = 512

X = tf.placeholder(tf.float32, [None, D])
Y_ = tf.placeholder(tf.float32, [None, K])

W0 = weight_variable([D, K1])
b0 = bias_variable([K1])

h_fc1 = tf.nn.relu(tf.matmul(X, W0) + b0)

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

def one_hot_encode(Y, k):
    assert(ml.dim(Y) == 1)
    n = Y.shape[0]
    assert n >= 1

    ret = np.zeros((n, k))
    ret[np.arange(Y.size), Y] = 1

    assert ret.shape == (n, k)
    assert ret[0][Y[0]] == 1

    return ret

def get_batch(filepath):
    global K
    dic = unpickle(filepath)
    # Only grabbing the first color channel for now
    return np.array(dic['data']), one_hot_encode(np.array(dic['labels']), K)

def ycrcb(x):
    a = np.zeros((32, 32, 3))
    a[:,:,0] = x[0:1024].reshape((32, 32))
    a[:,:,1] = x[1024:2048].reshape((32, 32))
    a[:,:,2] = x[2048:3072].reshape((32, 32))

    plt.imshow(a)
    return a
    # return cv2.cvtColor(a, cv2.cv.CV_BGR2YCrCb)


meta = unpickle('cifar/batches.meta')['label_names']
batches  = [get_batch('cifar/data_batch_{}'.format(i)) for i in xrange(1, 6)]
X_test, Y_test = get_batch('cifar/test_batch')

N_test = X_test.shape[0]

X_train, Y_train = zip(*batches)
X_train = np.concatenate(tuple(X_train))
Y_train = np.concatenate(tuple(Y_train))

N_train = X_train.shape[0]

print 'Done loading data'
print 'Training on {} samples. Testing on {} samples.'.format(N_train, N_test)


def mutate(X):
    return X

# All of the below should return vectors with elements between [0, 0.1]
dct_truncate_D = lambda x: jpeg.dct_truncate(x, D, 32, 32)
dct_truncate_D_split = lambda x: jpeg.dct_truncate(x, D / 3, 32, 32)
def dct_data(X):
    # Actually applies the DCT and truncates the vector

    # Code for DCT'ing three color channels separately
    X1 = X[:, :1024]
    X2 = X[:, 1024:2048]
    X3 = X[:, 2048:3072]

    X1 = np.apply_along_axis(dct_truncate_D_split, 1, X1)
    X2 = np.apply_along_axis(dct_truncate_D_split, 1, X2)
    X3 = np.apply_along_axis(dct_truncate_D_split, 1, X3)

    #X = minmax_scale(X)
    # return X
    # We have to squash the input between 0 and 1 to make the network converge
    X1 = np.apply_along_axis(minmax_scale, 1, X1)
    X2 = np.apply_along_axis(minmax_scale, 1, X2)
    X3 = np.apply_along_axis(minmax_scale, 1, X3)

    # Code to print out 0's and 1's
    # Xs = [X1, X2, X3]
    # for bigX in Xs:
    #     for i in xrange(32):
    #         out = ""
    #         for j in xrange(32):
    #             if abs(bigX[0][i * 32 + j]) < 0.01:
    #                 out += "0"
    #             else:
    #                 out += "1"
    #             out += ","
    #         print out

    # Recombine into one array
    X = [] 
    for i in xrange(len(X1)):
        append = []
        append.extend(X1[i])
        append.extend(X2[i])
        append.extend(X3[i])
        X.append(append)
    X = np.matrix(X) / 10.

    return np.matrix(X) / 10.

truncate_D = lambda x: jpeg.truncate(x, D)
def squash_data(X):
    truncated = np.apply_along_axis(truncate_D, 1, X)
    return jpeg.bitmap_to_normal(truncated, 2560)

def pca_data(X):
    scaled = jpeg.bitmap_to_normal(X, 2560)
    pca = PCA(n_components = D)
    return pca.fit_transform(scaled)


# ycrcb(X_train[0])
# ycrcb(X_train[1])
# plt.show()
# time.sleep(100)
# exit()
# np.apply_along_axis(ycrcb, 1, X_train)


mutate_data = dct_data

X_test = mutate_data(X_test)
X_train = mutate_data(X_train)

print 'Done mutating data. X_train.shape == {}'.format(X_train.shape)
# pdb.set_trace()

print X_test.shape, N_test, D

assert(X_test.shape == (N_test, D))
assert(Y_test.shape == (N_test, K))
assert(X_train.shape == (N_train, D))
assert(Y_train.shape == (N_train, K))

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
        print 'Train: ', accuracy.eval({X: X_train, Y_: Y_train, keep_prob: 1.0})
        print 'Test {}: {}'.format(i, accuracy.eval({X: X_test, Y_: Y_test, keep_prob: 1.0}))
    opt.run(feed_dict={X: batch[0], Y_: batch[1], keep_prob: 0.5})

print 'Train: ', accuracy.eval({X: X_train, Y_: Y_train, keep_prob: 1.0})
print 'Test {}: {}'.format(i, accuracy.eval({X: X_test, Y_: Y_test, keep_prob: 1.0}))
end = time.time()
print 'Time = {}'.format(end - start)
