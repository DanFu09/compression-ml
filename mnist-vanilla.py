import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import jpeg
import numpy as np

sess = tf.InteractiveSession()

## Model

# D = Input dimension, K = output dimension (# of classification categories)
D, K = 784, 10

W = tf.Variable(tf.zeros([D, K]))
b = tf.Variable(tf.zeros([K]))

X = tf.placeholder(tf.float32, [None, D])
Y_ = tf.placeholder(tf.float32, [None, K])

Y = tf.nn.softmax(tf.matmul(X, W) + b)

# We minimize the cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y), reduction_indices=[1]))
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cross_entropy)

# And use the accuracy as a measure of success
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

# Train
dct_truncate_D = lambda x: jpeg.dct_truncate(x, D)
def mutate(X):
    return X + 1
    #X = np.apply_along_axis(jpeg.normal_to_bitmap, 1, X)
    #X = np.apply_along_axis(dct_truncate_D, 1, X)
    return X
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print "Hello", mutate(mnist.test.images).shape

for i in xrange(1000):
    if i % 10 == 0:
        print accuracy.eval({X: mutate(mnist.test.images), Y_: mnist.test.labels})
    batch = mnist.train.next_batch(50)
    opt.run(feed_dict={X: mutate(batch[0]), Y_: batch[1]})

print accuracy.eval({X: mutate(mnist.test.images), Y_: mnist.test.labels})
