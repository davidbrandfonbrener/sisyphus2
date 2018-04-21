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
        # TODO better LSTM initialization
        # ----------------------------------
        self.N_concat = self.N_in + self.N_rec

        self.init_hidden_initializer = tf.random_normal_initializer(mean=0.1, stddev=0.01)
        self.init_cell_initializer = tf.random_normal_initializer(mean=0.1, stddev=0.01)

        self.W_f_initializer = tf.constant_initializer(
                0.1 * np.random.uniform(-1, 1, size=(self.N_concat, self.N_rec)))
        self.W_i_initializer = tf.constant_initializer(
                0.1 * np.random.uniform(-1, 1, size=(self.N_concat, self.N_rec)))
        self.W_c_initializer = tf.constant_initializer(
                0.1 * np.random.uniform(-1, 1, size=(self.N_concat, self.N_rec)))
        self.W_o_initializer = tf.constant_initializer(
                0.1 * np.random.uniform(-1, 1, size=(self.N_concat, self.N_rec)))

        self.b_f_initializer = tf.constant_initializer(0.0)
        self.b_i_initializer = tf.constant_initializer(0.0)
        self.b_c_initializer = tf.constant_initializer(0.0)
        self.b_o_initializer = tf.constant_initializer(0.0)

        # ----------------------------------
        # Tensorflow initializations
        # ----------------------------------

        self.init_hidden = tf.get_variable('init_hidden', [N_rec], initializer=b_f_initializer,
                                     trainable=True)
        self.init_cell = tf.get_variable('init_cell', [N_rec], initializer=b_f_initializer,
                                     trainable=True)

        self.W_f = tf.get_variable('W_f', [N_concat, N_rec],
                                        initializer=W_f_initializer,
                                        trainable=True)
        self.W_i = tf.get_variable('W_i', [N_concat, N_rec],
                                        initializer=W_i_initializer,
                                        trainable=True)
        self.W_c = tf.get_variable('W_c', [N_concat, N_rec],
                                        initializer=W_c_initializer,
                                        trainable=True)
        self.W_o = tf.get_variable('W_o', [N_concat, N_rec],
                                        initializer=W_o_initializer,
                                        trainable=True)

        self.b_f = tf.get_variable('b_f', [N_rec], initializer=b_f_initializer,
                                     trainable=True)
        self.b_i = tf.get_variable('b_f', [N_rec], initializer=b_f_initializer,
                                   trainable=True)
        self.b_c = tf.get_variable('b_f', [N_rec], initializer=b_f_initializer,
                                   trainable=True)
        self.b_o = tf.get_variable('b_f', [N_rec], initializer=b_f_initializer,
                                   trainable=True)




    def recurrent_timestep(self, rnn_in, hidden, cell):

        f = tf.nn.sigmoid(tf.matmul(self.W_f, tf.concat([hidden, rnn_in], 0))
                               + b_f)

        i = tf.nn.sigmoid(tf.matmul(self.W_i, tf.concat([hidden, rnn_in], 0))
                               + b_i)

        c = tf.nn.tanh(tf.matmul(self.W_c, tf.concat([hidden, rnn_in], 0))
                               + b_c)

        o = tf.nn.sigmoid(tf.matmul(self.W_o, tf.concat([hidden, rnn_in], 0))
                          + b_o)

        new_cell = f * cell + i * c

        new_hidden = o * tf.nn.sigmoid(new_cell)

        return new_hidden, new_cell




    def output_timestep(self, hidden):

        output = tf.matmul(self.W_out, hidden) + self.b_out

        return output



    def forward_pass(self):
        rnn_inputs = tf.unstack(self.x, axis=1)
        hidden = self.init_hidden
        cell = self.init_cell
        rnn_outputs = []
        rnn_states = []
        for rnn_input in rnn_inputs:
            hidden, cell = self.recurrent_timestep(rnn_input, hidden, cell)
            output = self.output_timestep(hidden)
            rnn_outputs.append(output)
            rnn_states.append(hidden)

        return tf.transpose(rnn_outputs, [1, 0, 2]), rnn_states
