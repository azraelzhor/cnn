import math
import numpy as np
import sys
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from utils.cnn_utils import *

def create_placeholders(n_W0, n_H0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_W0, n_H0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])
    return X, Y

def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    
    params = {"W1": W1, "W2": W2}
    return params

def forward_propagation(X, params):
    '''
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    '''
    W1 = params["W1"]
    W2 = params["W2"]

    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)

    return Z3

def compute_cost(Z3, Y):

    cost = tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y)
    cost = tf.reduce_mean(cost)
    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
        num_epochs=100, minibatch_size=64, verbose=True):
    
    ops.reset_default_graph() # rerun model without overwriting values
    tf.set_random_seed(1)
    seed = 3
    (m, n_W0, n_H0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    # create placeholders
    X, Y = create_placeholders(n_W0, n_H0, n_C0, n_y)

    # initialize parameters
    params = initialize_parameters()

    # build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, params)

    # add cost function to the tensorflow graph
    cost = compute_cost(Z3, Y)

    # define tensorflow optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    # start the session to compute the tensorflow graph
    with tf.Session() as sess:

        sess.run(init)

        # do the epoch loop
        for epoch in range(num_epochs):

            minibatch_cost = 0

            # change seed for each epoch
            seed = seed + 1

            # get batches of data
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            num_minibatches = int(m / minibatch_size)

            # do the iteration loop
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})                
                minibatch_cost += temp_cost / num_minibatches
            
            # print the cost every epoch
            if verbose == True:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                costs.append(minibatch_cost)

        # after training
        # plot
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iteration (per ten)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

        predict_op = tf.arg_max(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.arg_max(Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        print("Train accuracy: ", train_accuracy)
        print("Test accuracy: ", test_accuracy)

        return train_accuracy, test_accuracy, params

def run():

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T

    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    
    model(X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":
    tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(64, 64, 3, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
    print("Z3 = " + str(a))
    # run()
