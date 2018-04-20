from __future__ import print_function

import tensorflow as tf
import numpy as np
from time import time
import regularization


class RNN_model(object):
    def __init__(self, params):
        # ----------------------------------
        # Network sizes (tensor dimensions)
        # ----------------------------------
        N_in = self.N_in = params['N_in']
        N_rec = self.N_rec = params['N_rec']
        N_out = self.N_out = params['N_out']
        N_steps = self.N_steps = params['N_steps']
        N_batch = self.N_batch = params['N_batch']

        # ----------------------------------
        # Physical parameters
        # ----------------------------------
        self.dt = params['dt']
        self.tau = params['tau']
        self.alpha = self.dt / self.tau
        self.dale_ratio = params['dale_ratio']
        self.rec_noise = params['rec_noise']

        # ----------------------------------
        # Load weights path
        # ----------------------------------
        self.load_weights_path = params.get('load_weights_path', None)

        # ----------------------------------
        # Dale matrix
        # ----------------------------------
        dale_vec = np.ones(N_rec)
        if self.dale_ratio is not None:
            dale_vec[int(self.dale_ratio * N_rec):] = -1
            self.dale_rec = np.diag(dale_vec)
            dale_vec[int(self.dale_ratio * N_rec):] = 0
            self.dale_out = np.diag(dale_vec)
        else:
            self.dale_rec = np.diag(dale_vec)
            self.dale_out = np.diag(dale_vec)

        # ----------------------------------
        # Connectivity
        # ----------------------------------
        self.input_connectivity_mask = params.get('input_connectivity_mask', None)
        self.recurrent_connectivity_mask = params.get('recurrent_connectivity_mask', None)
        self.output_connectivity_mask = params.get('output_connectivity_mask', None)
        if self.input_connectivity_mask is None:
            self.input_connectivity_mask = np.ones((N_rec, N_in))
        if self.recurrent_connectivity_mask is None:
            self.recurrent_connectivity_mask = np.ones((N_rec, N_rec))
        if self.output_connectivity_mask is None:
            self.output_connectivity_mask = np.ones((N_out, N_rec))

        # ----------------------------------
        # MOVE THIS?????????????????????
        # regularization coefficients
        # ----------------------------------
        self.L1_in = params.get('L1_in', 0)
        self.L1_rec = params.get('L1_rec', 0)
        self.L1_out = params.get('L1_out', 0)

        self.L2_in = params.get('L2_in', 0)
        self.L2_rec = params.get('L2_rec', 0)
        self.L2_out = params.get('L2_out', 0)

        self.L2_firing_rate = params.get('L2_firing_rate', 0)
        self.sussillo_constant = params.get('sussillo_constant', 0)

        # ----------------------------------
        # Trainable features
        # ----------------------------------
        self.W_in_train = params.get('W_in_train', True)
        self.W_rec_train = params.get('W_rec_train', True)
        self.W_out_train = params.get('W_out_train', True)
        self.b_rec_train = params.get('b_rec_train', True)
        self.b_out_train = params.get('b_out_train', True)
        self.init_state_train = params.get('init_state_train', True)

        # ----------------------------------
        # Tensorflow input initializations
        # ----------------------------------
        self.x = tf.placeholder("float", [N_batch, N_steps, N_in])
        self.y = tf.placeholder("float", [N_batch, N_steps, N_out])
        self.output_mask = tf.placeholder("float", [N_batch, N_steps, N_out])

        # ------------------------------------------------
        # Define initializers for trainable variables
        # ------------------------------------------------
        if self.load_weights_path is None:
            init_state_initializer = tf.random_normal_initializer(mean=0.1, stddev=0.01)
            W_in_initializer = tf.constant_initializer(
                0.1 * np.random.uniform(-1, 1, size=(self.N_rec, self.N_in)))
            W_rec_initializer = tf.constant_initializer(self.initial_W())
            W_out_initializer = tf.constant_initializer(
                0.1 * np.random.uniform(-1, 1, size=(self.N_out, self.N_rec)))
            b_rec_initializer = tf.constant_initializer(0.0)
            b_out_initializer = tf.constant_initializer(0.0)
        else:
            print("Loading Weights")
            weights = np.load(self.load_weights_path)
            init_state_initializer = tf.constant_initializer(weights['init_state'])
            W_in_initializer = tf.constant_initializer(weights['W_in'])
            W_rec_initializer = tf.constant_initializer(weights['W_rec'])
            W_out_initializer = tf.constant_initializer(weights['W_out'])
            b_rec_initializer = tf.constant_initializer(weights['b_rec'])
            b_out_initializer = tf.constant_initializer(weights['b_out'])

            self.input_connectivity_mask = weights['input_Connectivity']
            self.recurrent_connectivity_mask = weights['rec_Connectivity']
            self.output_connectivity_mask = weights['output_Connectivity']

        self.init_state = tf.get_variable('init_state', [N_batch, N_rec],
                                          initializer=init_state_initializer)

        # ------------------------------------------------
        # Trainable variables:
        # Weight matrices and bias weights
        # ------------------------------------------------

        # Input weight matrix:
        self.W_in = \
            tf.get_variable('W_in', [N_rec, N_in],
                            initializer=W_in_initializer,
                            trainable=self.W_in_train)
        # Recurrent weight matrix:
        self.W_rec = \
            tf.get_variable(
                'W_rec',
                [N_rec, N_rec],
                initializer=W_rec_initializer,
                trainable=self.W_rec_train)
        # Output weight matrix:
        self.W_out = tf.get_variable('W_out', [N_out, N_rec],
                                     initializer=W_out_initializer,
                                     trainable=self.W_out_train)

        # Recurrent bias:
        self.b_rec = tf.get_variable('b_rec', [N_rec], initializer=b_rec_initializer,
                                     trainable=self.b_rec_train)
        # Output bias:
        self.b_out = tf.get_variable('b_out', [N_out], initializer=b_out_initializer,
                                     trainable=self.b_out_train)

        # ------------------------------------------------
        # Non-trainable variables:
        # Overall connectivity and Dale's law matrices
        # ------------------------------------------------

        # Recurrent Dale's law weight matrix:
        self.Dale_rec = tf.get_variable('Dale_rec', [N_rec, N_rec],
                                        initializer=tf.constant_initializer(self.dale_rec),
                                        trainable=False)
        # Output Dale's law weight matrix:
        self.Dale_out = tf.get_variable('Dale_out', [N_rec, N_rec],
                                        initializer=tf.constant_initializer(self.dale_out),
                                        trainable=False)

        # Connectivity weight matrices:
        self.input_Connectivity = tf.get_variable('input_Connectivity', [N_rec, N_in],
                                                  initializer=tf.constant_initializer(
                                                      self.input_connectivity_mask),
                                                  trainable=False)
        self.rec_Connectivity = tf.get_variable('rec_Connectivity', [N_rec, N_rec],
                                                initializer=tf.constant_initializer(
                                                    self.recurrent_connectivity_mask),
                                                trainable=False)
        self.output_Connectivity = tf.get_variable('output_Connectivity', [N_out, N_rec],
                                                   initializer=tf.constant_initializer(
                                                       self.output_connectivity_mask),
                                                   trainable=False)

        # --------------------------------------------------
        # Define the computation graph to calculate the loss
        # --------------------------------------------------
        self.predictions, self.states = self.compute_predictions()
        self.error = self.mean_square_error()
        self.loss = self.error + self.regularization()






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








    def train(self, generator,
              learning_rate=.001, training_iters=50000,
              batch_size=64, display_step=10, save_weights_step=100, save_weights_path=None,
              generator_function=None, training_weights_path=None):

        sess = tf.Session()


        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads = optimizer.compute_gradients(self.loss)
        clipped_grads = [(tf.clip_by_norm(grad, 1.0), var)
                         if grad is not None else (grad, var)
                         for grad, var in grads]
        optimize = optimizer.apply_gradients(clipped_grads)

        # run session
        sess.run(tf.global_variables_initializer())
        step = 1

        # time training
        t1 = time()

        # --------------------------------------------------
        # Training loop
        # --------------------------------------------------
        while step * batch_size < training_iters:
            batch_x, batch_y, output_mask = generator.next()
            sess.run(optimize, feed_dict={self.x: batch_x, self.y: batch_y, self.output_mask: output_mask})
            if step % display_step == 0:
                # --------------------------------------------------
                # Output batch loss
                # --------------------------------------------------
                loss = sess.run(self.loss,
                                feed_dict={self.x: batch_x, self.y: batch_y, self.output_mask: output_mask})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss))

                # --------------------------------------------------
                # Allow for curriculum learning
                # --------------------------------------------------
                if generator_function is not None:
                    generator = generator_function(loss, step)

            # --------------------------------------------------
            # Save intermediary weights
            # --------------------------------------------------
            if step % weight_save_step == 0:
                if training_weights_path is not None:
                    np.savez(training_weights_path + str(step), W_in=self.W_in.eval(session=sess),
                             W_rec=self.W_rec.eval(session=sess),
                             W_out=self.W_out.eval(session=sess),
                             b_rec=self.b_rec.eval(session=sess),
                             b_out=self.b_out.eval(session=sess),
                             init_state=self.init_state.eval(session=sess),
                             input_Connectivity=self.input_Connectivity.eval(session=sess),
                             rec_Connectivity=self.rec_Connectivity.eval(session=sess),
                             output_Connectivity=self.output_Connectivity.eval(session=sess))

            step += 1
        t2 = time()
        print("Optimization Finished!")

        # --------------------------------------------------
        # Save final weights
        # --------------------------------------------------
        if save_weights_path is not None:
            np.savez(save_weights_path, W_in=self.W_in.eval(session=sess),
                     W_rec=self.W_rec.eval(session=sess),
                     W_out=self.W_out.eval(session=sess),
                     b_rec=self.b_rec.eval(session=sess),
                     b_out=self.b_out.eval(session=sess),
                     init_state=self.init_state.eval(session=sess),
                     input_Connectivity=self.input_Connectivity.eval(session=sess),
                     rec_Connectivity=self.rec_Connectivity.eval(session=sess),
                     output_Connectivity=self.output_Connectivity.eval(session=sess))
            print("Model saved in file: %s" % save_weights_path)


        sess.close()

        return (t2 - t1)




