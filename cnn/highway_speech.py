import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape, variable_name):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init, name=variable_name)
    
def bias_variable(shape, variable_name):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init, name=variable_name)
    
def highway(x, size, activation, carry_bias=-1.0):
    W_T = weight_variable([size, size], "weight_transform")
    b_T = bias_variable([size], "bias_transform")
    
    W = weight_variable([size, size], "weight")
    b = bias_variable([size], "bias")

    T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name="transform_gate")
    H = activation(tf.matmul(x, W) + b, name="activation")
    C = tf.sub(1.0, T, name="carry_gate")

    y = tf.add(tf.mul(H, T), tf.mul(x, C), "y")
    return y

def dnn(x, in_size, out_size, activition):
    W = weight_variable([in_size, out_size], "weight")
    b = bias_variable([out_size], "bias")
  
    return activation(tf.matmul(x, W) + b)
  
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

#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# placeholder for input variables
x = tf.placeholder(tf.float32, [None, 784])/255
y = tf.placeholder(tf.float32, [None, 10])

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
