import tensorflow as tf
import os
import shutil
import numpy as np
#from math_ops import combinations

# Function for the first trap layer, unused right now.
def trap_layer(input_tensor, group_size, output_group_size, batch_size=1):
    """
    :param input_tensor: Tensor to be treated
    :param group_size: Size of the group of variables in the trap function
    :param output_group_size: Amount of neurons outputed from each group
    :param batch_size: size of the batch
    :return: Tensor representing a layer
    """
    ind_len = int(input_tensor.shape[1])
    groups = int(ind_len/group_size)
    # The weights are organized in a "flat matrix". Convenient multiplication is after applied
    weights = tf.Variable(tf.random_uniform([ind_len, output_group_size]), name="Weights")
    biases = tf.Variable(tf.random_uniform([output_group_size*groups]), name="Biases")

    layer = None

    for group in range(groups):
        inp = tf.reshape(input_tensor[:, group*group_size:(group+1)*group_size], (-1, group_size))  # Select segment of the layer
        ws = tf.reshape(weights[group*group_size:(group+1)*group_size, :], (group_size, output_group_size))  # Select segment of the weights
        if layer is not None:
            layer = tf.concat((layer, tf.matmul(inp, ws)), axis=1)
        else:
            layer = tf.matmul(inp, ws)

    return layer+biases, weights, biases


"""
This function takes a tensor as an input and returns a layer conveniently adapted to the 
variable groups provided. That layer will contain a certain amount of neurons per ADF.
For example, if three ADFs are encapsulated in the fitness functionand the variable sets 
used by each one are [x1, x2, x3], [x1, x5, x6], and [x2, x4], and two neurons need to be
created for each ADF, 16 weights are used in the layer, and the multiplication is performed
in a way such that each pair of neurons get non-zero signal only from those inputs (variables)
present in their parameters.

This is a generalized version of the previous function.
"""

def ad_hoc_layer(input_tensor, variable_groups, output_group_size, weights, biases, batch_size):
    """
    :param input_tensor: value over which the layer is built. Its natural values are the initial solution.
    :param variable_groups: List of lists. Each sublist in position i contains the indices of the variables
    used by function i.
    :param output_group_size: Amount of neurons in the new layer for each subfunction.
    :return:
    """

    i = 0
    layer = None
    for sub_group in variable_groups:
        aux = tf.transpose(input_tensor)
        variable_group = tf.gather(aux, sub_group)
        variable_group = tf.transpose(variable_group)
        #print(variable_group)
        #variable_group = tf.gather_nd(input_tensor, combinations(range(batch_size), sub_group))

        #variable_group = tf.reshape(variable_group, [-1, len(sub_group)])
        #print(variable_group)
        # The previous two lines select, from the previous layer (the initial in this case),
        # the neurons (variables) that will affect each neuron in the following layer.
        # I think there will be a more sophisticated way of doing this, I'm not sure how
        # optimizable this is in the GPU, by tensorflow

        for out_neuron in range(output_group_size):
            ln = int(variable_group.shape[1])
            ws = tf.gather(weights, np.arange(i, i+ln))
            ws = tf.reshape(ws, (ws.shape[0], 1))

            if layer is not None:
                layer = tf.concat((layer, tf.matmul(variable_group, ws)), axis=1)
            else:
                layer = tf.matmul(variable_group, ws)

            i += ln

    layer = layer + biases
    return layer


"""
Function where all the TF variables are initialized. Takes as input the structure of the net
and returns its TF components
"""


def initialize_variables(input_size, output_size, h_neurons, batch_size, learning_rate, sigma, variable_groups, structure, delete=False):
    """
    :param input_size: Size of the input for the net
    :param h_neurons0: Amount of neurons in the first hidden layer
    :param h_neurons1: Amount of neurons in the second hidden layer
    :param batch_size: Instances for each training epoch
    :param learning_rate: Multiplier to be applied to the gradients
    :param sigma: Sigma parameter of the logistic function for the input binarization
    :param variable_groups: Groups of variables in the ADF (for the ad_hoc layer creation)
    :param delete: Boolean. Whether the training is going to be performed from the start (True) or not (False)
    :return: All the components of the net
    """
    log = "Board/"  # Folder where model checkpoints and data records are saved
    if os.path.exists(log) and delete:
        shutil.rmtree(log)

    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}))

    # ############# Forward net ############# #

    # Input & Output
    x = tf.placeholder(tf.float32, [None, input_size], name="x")  # x will contain the data batches
    y = tf.placeholder(tf.float32, [None, output_size], name="y")  # y will contain the label batches
    # In a more advanced version, this should be included, to "binarize" the individuals
    clipped_x = x  # tf.div(1.0, tf.add(1.0, tf.exp(tf.subtract(tf.multiply(-sigma, x), -sigma/2))))

    # Weights & Biases
    if structure == 0:
        weights = tf.get_variable("W", shape=[input_size, h_neurons], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("B", shape=[h_neurons], initializer=tf.contrib.layers.xavier_initializer())
        layer0 = tf.nn.relu(tf.matmul(clipped_x, weights) + biases)
    else:
        output_group_size = int(h_neurons/len(variable_groups))
        ln = [x*output_group_size for subgroup in variable_groups for x in subgroup]
        weights = tf.get_variable("W", shape=[len(ln)*output_group_size], initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("B", shape=[1, output_group_size*len(variable_groups)], initializer=tf.contrib.layers.xavier_initializer())
        layer0 = tf.nn.relu(ad_hoc_layer(clipped_x, variable_groups, output_group_size, batch_size=batch_size, weights=weights, biases=biases))

    hidden_weights0 = tf.get_variable("hW1", shape=[h_neurons, output_size], initializer=tf.contrib.layers.xavier_initializer())

    hidden_biases0 = tf.get_variable("hB1", shape=[output_size], initializer=tf.contrib.layers.xavier_initializer()) #tf.Variable(tf.random_normal([h_neurons1]), name="Bias")

    net = tf.nn.sigmoid(tf.matmul(layer0, hidden_weights0) + hidden_biases0)

    # Cost function and Optimizer

    loss = tf.reduce_mean(tf.pow(net-y, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # ############# Tensorboard visualization ############# #

    sum_layer0 = tf.summary.histogram("layer", layer0)
    sum_weights = tf.summary.histogram("weights", weights)
    sum_hidden_weights0 = tf.summary.histogram("hidden_weights", hidden_weights0)
    sum_hidden_biases0 = tf.summary.histogram("hidden_biases", hidden_biases0)
    sum_biases = tf.summary.histogram("biases", biases)

    sum_clipped_x = tf.summary.histogram("clipped_x", clipped_x)

    sum_net = tf.summary.histogram("net", net)
    sum_loss = tf.summary.scalar("loss", loss)

    sum_ford = tf.summary.merge([sum_clipped_x, sum_weights, sum_hidden_weights0, sum_layer0, sum_biases, sum_hidden_biases0, sum_net, sum_loss], name="Forward Summary")

    # To save the model

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(log + "Graph")
    writer.add_graph(sess.graph)

    return sess, x, y, weights, hidden_weights0, biases, hidden_biases0, net, loss, optimizer, sum_ford, saver, writer, log


# These functions are necessary for the backdrive if the convolutional layer is used. They return
# The weights and biases respectively.
def aux1(*args, **kwargs):
    gr = tf.get_default_graph()
    return gr.get_tensor_by_name('conv1d/kernel:0')


def aux2(*args, **kwargs):
    gr = tf.get_default_graph()
    return gr.get_tensor_by_name('conv1d/bias:0')


# Analogous to initialize_variables above. Only that this is not as complex, since the tuning is
# always performed from the start.
def initialize_back_variables(sess, input_size, output_size, h_neurons, sigma, variable_groups, structure, number):

    log = "BackBoard/"  # Folder where log is stored
    if os.path.exists(log):
        shutil.rmtree(log)

    # ############# Backwards Net ############# #

    ind = tf.Variable(tf.random_uniform((number, input_size)))

    ind_initializer = tf.variables_initializer([ind])

    """Old initialization
    trainables = np.random.choice(range(input_size), int(np.ceil(input_size/2)), replace=False)
    values = np.random.choice([0, 1], int(input_size/2))
    trainable = []
    #print(trainables, values)

    for i in range(input_size):
        if i not in trainables:
            values, pop = values[:-1], values[-1]
            ind += [tf.Variable(float(pop))]
        else:
            ind += [tf.Variable(tf.random_uniform((1,)))]
            trainable += [ind[-1]]
    ind_initializer = tf.variables_initializer(ind)
    ind = tf.concat([ind], axis=0)
    """



    # Input & Output

    #ind = tf.Variable(np.reshape(np.array([.5]*input_size), (1, input_size)), name="Back_input", dtype="float")
    clipped_ind = tf.clip_by_value(ind, 0.0, 1.0)
    #clipped_ind = tf.div(1.0, tf.add(1.0, tf.exp(tf.subtract(tf.multiply(-sigma, tf.reshape(tf.concat([ind], axis=0), (1, -1))), -sigma/2))))
    target = tf.placeholder("float32", [None, output_size], name="Target")

    # Weights & Biases

    if structure == 0:
        back_weights = tf.placeholder("float32", [input_size, h_neurons], name="back_hidden_weights")
        back_biases = tf.placeholder("float32", [h_neurons], name="back_hidden_biases")
        back_layer0 = tf.nn.relu(tf.matmul(clipped_ind, back_weights) + back_biases)
    else:
        output_group_size = int(h_neurons/len(variable_groups))
        ln = [x*output_group_size for subgroup in variable_groups for x in subgroup]
        back_weights = tf.placeholder("float32", [len(ln)*output_group_size], name="back_weights")
        back_biases = tf.placeholder("float32", [1, output_group_size*len(variable_groups)], name="back_biases")
        back_layer0 = tf.nn.relu(ad_hoc_layer(clipped_ind, variable_groups, int(h_neurons/len(variable_groups)), batch_size=1, weights=back_weights, biases=back_biases))

    back_hidden_weights = tf.placeholder("float32", [h_neurons, output_size], name="back_hidden_weights")

    back_hidden_biases = tf.placeholder("float32", [output_size], name="back_hidden_biases")

    back_net = tf.nn.sigmoid(tf.matmul(back_layer0, back_hidden_weights) + back_hidden_biases)

    # Cost function and Optimizer
    back_loss = tf.pow(back_net-target, 2)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    grads = optimizer.compute_gradients(loss=back_loss, var_list=[ind])
    apply_grads = optimizer.apply_gradients(grads)

    sum_back_layer0 = tf.summary.histogram("back_layer", back_layer0)
    sum_clipped_ind = tf.summary.histogram("clipped_ind", clipped_ind)
    sum_ind = tf.summary.histogram("ind", ind)

    sum_back_net = tf.summary.histogram("back_net", back_net)
    sum_back_loss = tf.summary.scalar("back_loss", back_loss)

    sess.run(tf.global_variables_initializer())
    sum_back = tf.summary.merge([sum_clipped_ind, sum_back_net, sum_back_layer0, sum_back_loss, sum_ind], name="Backwards Summary")
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(log + "Graph")
    writer.add_graph(sess.graph)
    return ind, ind_initializer, clipped_ind, target, back_weights, back_hidden_weights, back_biases, back_hidden_biases, back_net, back_loss, apply_grads, grads, optimizer, sum_back, saver, writer, log


def close(sess):
    tf.reset_default_graph()
    sess.close()


if __name__ == '__main__':
    # Test the ad_hoc layer
    variable_groups = [[0, 1, 2], [0, 4, 5], [1, 3]]
    output_group_size = int(3 / len(variable_groups))
    ln = [x * output_group_size for subgroup in variable_groups for x in subgroup]
    #weights = tf.Variable(tf.random_uniform([len(ln)*output_group_size]), name="Weights")
    weights = tf.Variable([1.]*(len(ln)*output_group_size))
    #biases = tf.Variable(tf.random_uniform([1, output_group_size*len(variable_groups)]), name="Biases")
    biases = tf.Variable([[0.]*(output_group_size*len(variable_groups))])
    ad_hc = ad_hoc_layer(input_tensor=tf.Variable([[1.,2.,3.,4.,5.,6.], [7.,8.,9.,10.,11.,12.]]), variable_groups=variable_groups, output_group_size=output_group_size, weights=weights, biases=biases, batch_size=2)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    val = sess.run(ad_hc)
    print(val)
