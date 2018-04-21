from backend.rnn import RNN
import tensorflow as tf


class LSTM(RNN):

    def __init__(self, params):
        # ----------------------------------
        # Call RNN constructor
        # ----------------------------------
        super(LSTM, self).__init__(params)

        # ----------------------------------
        # Add new variables for gates
        # ----------------------------------



    def recurrent_timestep(self, rnn_in, state):

        pass


    def output_timestep(self, new_state):

        pass


    def forward_pass(self):

        pass