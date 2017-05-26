import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)
    
def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)
    
def conv2d(x, W):
    # strides=[1, x_move, y_move, 1], why first and last has to be 1
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    
def max_pool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
def activition(x):
    # Chosse activition function: 
    # - tf.nn.relu
    # - tf.nn.softplus
    # - tf.nn.dropout (when overfit)
    # - tf.sigmoid
    # - tf.tanh
    return tf.nn.relu(x)
    
def compute_cost(y, prediction):
    # Use the cost function to calculate cost sum
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    # calculate mean
    return tf.reduce_mean(cost)
    
def minimize_cost(cost):
    # Choose optimizer(learning rate)
    # - SGD 
    # - AdamOptimizer (typically for neural nets)
    return tf.train.AdamOptimizer(1e-4).minimize(cost)
    
# Temporary
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x: v_xs, y: v_ys, keep_prob: 1})
    return result

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# placeholder for input variables
x = tf.placeholder(tf.float32, [None, 784])/255
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x, [-1, 28, 28, 1])

# params for first conv net patch 5x5
W1 = weight_variable([5, 5, 1, 32])
b1 = bias_variable([32])
h1 = activition(conv2d(x_image, W1) + b1)
h1_pool = max_pool2d(h1)

# params for second conv net patch 5x5
W2 = weight_variable([5, 5, 32, 64])
b2 = bias_variable([64])
h2 = activition(conv2d(h1_pool, W2) + b2)
h2_pool = max_pool2d(h2)

# Fully connected layer 1, since the incoming input is 2d, we flatten it to 1d
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h2_pool_flat = tf.reshape(h2_pool, [-1, 7*7*64])
h_fc1 = activition(tf.matmul(h2_pool_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer 2, output the prediction classes
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

train_step = minimize_cost(compute_cost(y, prediction))

# Start session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    print(batch_xs.shape, batch_ys.shape)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
    if (i % 50 == 0):
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
