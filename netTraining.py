import os
import numpy as np
import tensorflow as tf
import pandas as pd
from structure import initialize_variables

"""
This function receives inputs and outputs intended for the network training, a desired batch size, and
the index of the last entry used to train the net in the last epoch. It returns a new batch starting
from the first unused index in the previous epoch, with the desired size
"""
def batch(x, y, size, i):
    """
    :param x: Poplation; set of solutions intended to be fed to the net in the input
    :param y: Fitness scores of the population, intended to be fed to the net in the output
    :param size: Size of the batch desired
    :param i: Index of the last solution used in the last epoch
    :return: The index of the last solution in the batch (to be provided to this same
             function in the next epoch, the solutions in the actual batch, and their
             respective fitness scores
    """

    if i + size > x.shape[0]:  # In case there are not enough solutions before the end of the array
        index = i + size-x.shape[0]  # Select all the individuals until the end and restart
        return index, np.concatenate((x[i:, :], x[:index, :])), np.concatenate((y[i:], y[:index]))
    else:  # Easy case
        index = i+size
        return index, x[i:index, :], y[i:index]

"""
This function is the responsible for training the net. First checks if the training is performed over
an already existing model. If this is not the case, this function initializes a new model. If the
model already exists, this function expects all its components as parameters. Then,trains the net 
until the error criterion is met. Finally, if the model was nonexistent previous to the training, 
return all the components in the model.
"""


def train(net_input, net_output, display_step, save_step, start=False, sess=None, loss=None, optimizer=None, x=None, y=None, sum_ford=None, writer=None, saver=None, log=None, net=None, epoch=1, sigma=None, hidden_neurons=None, batch_size=None, learning_rate=None, groups=None, structure=None, init=False):
    """
    :param net_input: Population of solutions
    :param net_output: Fitness scores of the solutions in the population
    :param display_step: The frequency results are saved to be displayed in tensorboard
    :param save_step: Frequency of model saving
    :param start: Boolean. Whether the training starts from an existing model (False) or not (True)

    In case the net is already existent, the following parameters are expected:

    :param sess: TF session
    :param loss: Loss function over which the net is trained
    :param optimizer: Optimization method used to train the net
    :param x: TF placeholder where the solutions are to be introduced
    :param y: TF placeholder where the expected outputs of the network are placed
    :param sum_ford: Summary of variables to be shown in tensorboard
    :param writer: TF writer variable to save partial results for tensorboard
    :param saver: TF saver variable to save the partial models
    :param log: Directory where the models and tensorboard results are stored
    :param net: Output of the net
    :param epoch: Epoch in which the previous model stopped training

    If not, these are expected:

    :param sigma: Sigma parameter of the logistic function applied to the individual at the beginning
    :param hidden_neurons: Amount of neurons in the first hidden layer
    :param learning_rate: Learning rate applied to the optimization method
    :param batch_size: Size of the batch used in each training epoch
    :param groups: Groups of variables in the ADF
    :param structure: Whether the human knowledge is used or not
    :return: Trains the net. Additionally, in case the net was non-existent, returns all its components
    """

    preds = pd.DataFrame()
    #preds[0] = net_output
    if init:
        sess.run(tf.global_variables_initializer())
    if start:  # If no net has been created yet, create it
        sess, x, y, weights,  hidden_weights, biases, hidden_biases, net, loss, optimizer, sum_ford, saver, writer, log = initialize_variables(input_size=net_input.shape[1], h_neurons=hidden_neurons, batch_size=batch_size, learning_rate=learning_rate, sigma=sigma, variable_groups=groups, structure=structure, delete=start)

    losses = list(range(int(net_input.shape[0]/batch_size)))  # Error FIFO
    i = 0
    iter_limit = 10000

    while epoch < iter_limit:
        i, batch_x, batch_y = batch(net_input, net_output, batch_size, i)  # New batch creation
        losses = losses[1:]  # Delete last error
        _, ls = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y})
        if epoch % display_step == 0:  # If necessary, save results to tensorboard
            print("Iter:", epoch, "Loss:", ls)
            #kl = sess.run(sum_KL, feed_dict={x: net_input, y: net_output})

            #preds[preds.shape[1]] = sess.run(net, feed_dict={x: net_input[:50]})
            #writer.add_summary(s, epoch)  # Write summary
            #writer.add_summary(kl, epoch)
            #print(pred)

        if epoch % save_step == 0:  # If necessary, save the model

            pass  # saver.save(sess, os.path.join(log, "model.ckpt"), epoch)

        losses.append(ls)  # Store new error
        #print(epoch)
        epoch += 1
    preds = preds.transpose()
    if saver is not None:
        saver.save(sess, os.path.join(log, "result.ckpt"), epoch)  # Save optimum error

    if start:  # If the net was not created previous to this function call, return its components
        return sess, x, y, weights, hidden_weights, biases, hidden_biases, net, loss, optimizer, sum_ford, saver, writer, log, preds
    else:
        return preds


"""
This function takes the parameters (weights and biases) evolved in the net training phase and uses
them to evolve new solutions, that (should) fulfill the required fitness score.
"""


def back_drive_net(sess, convergence, tgt, target, apply, perf_loss, back_weights, back_hidden_weights, back_biases, back_hidden_biases, w, h_w, b, h_b, iter_lim, clip, net):
    """
    :param sess: TF session
    :param convergence: Expected accuracy of the "created" solution
    :param tgt: Expected fitness score of the solution to be "created"
    :param target: TF placeholder where tgt is expected
    :param apply: Gradient application in the solution, TF operation
    :param perf_loss: loss function used for the new solution tuning
    :param back_weights: TF placeholder where the weights of the first layer are expected
    :param back_hidden_weights: TF placeholder where the weights of the first hidden layer are expected
    :param back_biases: TF placeholder where the biases of the first layer are expected
    :param back_hidden_biases: TF placeholder where the biases of the first hidden layer are expected

    The following variables are supposed to be those evolved in the net training phase

    :param w: weights of the first layer of the network
    :param h_w: weights of the first hidden layer of the network
    :param b: biases of the first layer of the network
    :param h_b: biases of the first hidden layer of the network
    :param iter_lim: Iteration limit for the development of the new solution
    :return: Nothing, the solution is evolved, though
    """
    it = 0
    ls = 1
    while it < iter_lim and np.max(ls) > 0.001:  # If the error is too large and we are
        it += 1
        ls, p = sess.run([perf_loss, net], feed_dict={target: tgt, back_hidden_biases: h_b, back_hidden_weights: h_w, back_weights: w, back_biases: b})
        #print(it, ls)# within the limit, continue training
        _, ls, p = sess.run([apply, perf_loss, net], feed_dict={target: tgt, back_hidden_biases: h_b, back_hidden_weights: h_w, back_weights: w, back_biases: b})

    #if it == iter_lim:
        #print("Limit surpassed:", iter_lim)
