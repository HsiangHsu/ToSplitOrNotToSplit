import tensorflow as tf
import numpy as np
import scipy as sp

num_neuron = 30
lr = 1e-2
epoch = 1000

def compute_divergence(X1, X0):
    dx = X0.shape[1]
    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    G_W1 = tf.Variable(xavier_init([dx, num_neuron]), name='G_W1')
    G_b1 = tf.Variable(tf.zeros(shape=[num_neuron]), name='G_b1')
    G_W2 = tf.Variable(xavier_init([num_neuron, num_neuron]), name='G_W2')
    G_b2 = tf.Variable(tf.zeros(shape=[num_neuron]), name='G_b2')
    G_W3 = tf.Variable(xavier_init([num_neuron, num_neuron]), name='G_W3')
    G_b3 = tf.Variable(tf.zeros(shape=[num_neuron]), name='G_b3')
    G_W4 = tf.Variable(xavier_init([num_neuron, 1]), name='G_W4')
    G_b4 = tf.Variable(tf.zeros(shape=[1]), name='G_b4')
    theta_G = [G_W1, G_b1, G_W2, G_b2, G_W3, G_b3, G_W4, G_b4]

    def g_approx(data):
        fc1 = tf.nn.relu(tf.matmul(data, G_W1) + G_b1)
        fc2 = tf.nn.relu(tf.matmul(fc1, G_W2) + G_b2)
        fc3 = tf.nn.relu(tf.matmul(fc2, G_W3) + G_b3)
        # fc4 = tf.nn.relu(tf.matmul(fc3, G_W4) + G_b4)
        # g = tf.matmul(fc4, G_W5) + G_b5
        g = tf.matmul(fc3, G_W4) + G_b4

        clip_min = np.float32(-.5)
        clip_max = np.float32(.5)
        g_clip = tf.clip_by_value(g, clip_min, clip_max)
        return g_clip

    def tv_divergence(data1, data2):
        M1 = g_approx(data1)
        M2 = g_approx(data2)
        sup_loss = tf.reduce_mean(M1) - tf.reduce_mean(M2)
        return sup_loss

    # Initiate the session for training
    sess = tf.InteractiveSession()

    # Placeholders for data from two populations
    data1 = tf.placeholder(tf.float32, [None, dx])
    data2 = tf.placeholder(tf.float32, [None, dx])

    # Compute mutual information
    TV = tv_divergence(data1, data2)
    TV_loss = -1*TV
    solver_TV = tf.train.AdagradOptimizer(lr).minimize(TV_loss, var_list = theta_G)

    # Initialization
    tf.global_variables_initializer().run()


    # Training
    # print('epoch\t loss')
    for i in range(epoch):
        _, current_loss = sess.run([solver_TV, TV_loss], feed_dict={data1: X1, data2: X0})
        # if i % 100 == 0:
        #     print('{}\t {:.8f}'.format(i, -current_loss))


    sess.close()

    return -current_loss
