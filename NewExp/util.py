import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import scipy as sp

# create simple Feed Foreword NN for experiments
def SimpleNet(inputShape, structure = [40, 40, 20, 15], name="simpleNet"):
    """
    Creates a simple feed forward neural network.

    Input:  the standard input will have 1 dimension, You could replace this to call any
            other constructor of a neural network. Structure defines the shape. The final
            layer will be d

    Output: Three variables that can be used to control the network:
            - x_input: input to the NN
            - f_out: output at the final layer
            - keepProb: dropout probability
    """

    # create name scope
    with tf.variable_scope(name):

        # creat input to network
        x_input = tf.placeholder(tf.float32,shape=[None,inputShape])

        # initialize with no dropout
        keepProb = tf.placeholder_with_default(1.0,[])

        # create list of intermediate outputs
        midOutputs = []

        # current layer being built
        layer = 0

        # total number of layers
        numLayers = len(structure)

        # create input layer
        with tf.variable_scope("input_layer", reuse=tf.AUTO_REUSE):
            numGates = structure[0]
            weights = tf.get_variable("weights",[inputShape,numGates],
                                      initializer=tf.truncated_normal_initializer(stddev=.1))
            bias = tf.get_variable("biases",[numGates],initializer=tf.constant_initializer(.1))

            output = tf.nn.tanh(tf.matmul(x_input,weights)+bias)
            # output = tf.nn.relu(tf.matmul(x_input,weights)+bias)
            midOutputs.append(tf.nn.dropout(output,keepProb))
        layer+=1

        # create intermediate layers
        for i in range(1,numLayers-1):
            with tf.variable_scope("middle_layer_"+str(i), reuse=tf.AUTO_REUSE):
                numGates = structure[i]
                numInputs = structure[i-1]

                weights = tf.get_variable("weights",[numInputs,numGates],
                                      initializer=tf.truncated_normal_initializer(stddev=.1))
                bias = tf.get_variable("biases",[numGates],initializer=tf.constant_initializer(.1))

                output = tf.nn.tanh(tf.matmul(midOutputs[layer-1],weights)+bias)
                # output = tf.nn.relu(tf.matmul(midOutputs[layer-1],weights)+bias)
                # output = tf.nn.sigmoid(tf.matmul(midOutputs[layer-1],weights)+bias)
                midOutputs.append(tf.nn.dropout(output,keepProb))

            layer+=1

        # create output layer
        with tf.variable_scope("output_layer", reuse=tf.AUTO_REUSE):
            numGates = structure[-1]
            numInputs = structure[-2]
            weights = tf.get_variable("weights",[numInputs,numGates],
                                      initializer=tf.truncated_normal_initializer(stddev=.1))
            bias = tf.get_variable("biases",[numGates],initializer=tf.constant_initializer(.1))

            # no relu in output
            # output = tf.nn.sigmoid(tf.matmul(midOutputs[layer-1],weights)+bias)
            output = tf.nn.relu(tf.matmul(midOutputs[layer-1],weights)+bias)
            # output = tf.nn.tanh(tf.matmul(midOutputs[layer-1],weights)+bias)

            #f_clip = tf.clip_by_value(final_output,-10,10)
    # return values
    return x_input, output

# create simple Feed Foreword NN for experiments
def H_Net(x_input, inputShape, structure = [40, 40, 20, 15], name="simpleNet"):
    """
    Creates a simple feed forward neural network.

    Input:  the standard input will have 1 dimension, You could replace this to call any
            other constructor of a neural network. Structure defines the shape. The final
            layer will be d

    Output: Three variables that can be used to control the network:
            - x_input: input to the NN
            - f_out: output at the final layer
            - keepProb: dropout probability
    """

    # create name scope
    with tf.variable_scope(name):

        # creat input to network
        # x_input = tf.placeholder(tf.float32,shape=[None,inputShape])

        # initialize with no dropout
        keepProb = tf.placeholder_with_default(1.0,[])

        # create list of intermediate outputs
        midOutputs = []

        # current layer being built
        layer = 0

        # total number of layers
        numLayers = len(structure)

        # create input layer
        with tf.variable_scope("input_layer", reuse=tf.AUTO_REUSE):
            numGates = structure[0]
            weights = tf.get_variable("weights",[inputShape,numGates],
                                      initializer=tf.truncated_normal_initializer(stddev=.1))
            bias = tf.get_variable("biases",[numGates],initializer=tf.constant_initializer(.1))

            output = tf.nn.tanh(tf.matmul(x_input,weights)+bias)
            # output = tf.nn.relu(tf.matmul(x_input,weights)+bias)
            midOutputs.append(tf.nn.dropout(output,keepProb))
        layer+=1

        # create intermediate layers
        for i in range(1,numLayers-1):
            with tf.variable_scope("middle_layer_"+str(i), reuse=tf.AUTO_REUSE):
                numGates = structure[i]
                numInputs = structure[i-1]

                weights = tf.get_variable("weights",[numInputs,numGates],
                                      initializer=tf.truncated_normal_initializer(stddev=.1))
                bias = tf.get_variable("biases",[numGates],initializer=tf.constant_initializer(.1))

                output = tf.nn.tanh(tf.matmul(midOutputs[layer-1],weights)+bias)
                # output = tf.nn.relu(tf.matmul(midOutputs[layer-1],weights)+bias)
                # output = tf.nn.sigmoid(tf.matmul(midOutputs[layer-1],weights)+bias)
                midOutputs.append(tf.nn.dropout(output,keepProb))

            layer+=1

        # create output layer
        with tf.variable_scope("output_layer", reuse=tf.AUTO_REUSE):
            numGates = structure[-1]
            numInputs = structure[-2]
            weights = tf.get_variable("weights",[numInputs,numGates],
                                      initializer=tf.truncated_normal_initializer(stddev=.1))
            bias = tf.get_variable("biases",[numGates],initializer=tf.constant_initializer(.1))

            # no relu in output
            # output = tf.nn.sigmoid(tf.matmul(midOutputs[layer-1],weights)+bias)
            output = tf.nn.relu(tf.matmul(midOutputs[layer-1],weights)+bias)
            # output = tf.nn.tanh(tf.matmul(midOutputs[layer-1],weights)+bias)

            #f_clip = tf.clip_by_value(final_output,-10,10)
    # return values
    return output
