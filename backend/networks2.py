from __future__ import print_function

import tensorflow as tf
import numpy as np
from time import time
import regularization


class RNN_model(object):
    def __init__(self, params):


    def rnn_step(self, rnn_in, state):

        pass


    def rnn_output(self, new_state):

        pass


    def forward_pass(self):

        pass


    def mean_square_error(self):
        return tf.reduce_mean(tf.square(self.output_mask * (self.predictions - self.y)))


    def reg_loss(self):
        return self.mean_square_error() + regularization.regularization(self)


    def train(self, sess, generator,
              learning_rate=.001, training_iters=50000,
              batch_size=64, display_step=10, weight_save_step=100, save_weights_path=None,
              generator_function=None, training_weights_path=None):

        return


    def test(self, sess, rnn_in, weights_path=None):

        return






class Basic(RNN_model):

    def rnn_step(self, rnn_in, state):

        pass


    def rnn_output(self, new_state):

        pass


    def forward_pass(self):

        pass





class Basic_scan(RNN_model):

    def rnn_step(self, rnn_in, state):

        pass


    def rnn_output(self, new_state):

        pass


    def forward_pass(self):

        pass







class LSTM(RNN_model):

    def rnn_step(self, rnn_in, state):

        pass


    def rnn_output(self, new_state):

        pass


    def forward_pass(self):

        pass