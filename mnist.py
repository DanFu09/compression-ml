import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.InteractiveSession()

# Model
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

X = tf.placeholder(tf.float32, [None, 784])
Y_ = tf.placeholder(tf.float32, [None, 10])

Y = tf.nn.softmax(tf.matmul(X, W) + b)

# We minimize the cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y), reduction_indices=[1]))
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cross_entropy)

# And use the accuracy as a measure of success
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

# Train
mutate = lambda x: x + 0.1
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
for i in xrange(1000):
    if i % 10 == 0:
        print accuracy.eval({X: mutate(mnist.test.images), Y_: mnist.test.labels})
    batch = mnist.train.next_batch(50)
    opt.run(feed_dict={X: mutate(batch[0]), Y_: batch[1]})

print accuracy.eval({X: mutate(mnist.test.images), Y_: mnist.test.labels})
