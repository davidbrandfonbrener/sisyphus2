from __future__ import print_function

import tensorflow as tf
import numpy as np
from time import time

# Lets make sure to keep things object-oriented,
# so that all future networks we build will extend
# the Model class below

# This will mean (in the future) making Model less specific so
# that future networks will "fill in the specifics" instead
# i.e. we can make a denseRNN, a sparseRNN, a denseCNN etc


class Model(object):
    def __init__(self, params):

        # Network sizes (tensor dimensions)
        N_in    = self.N_in       = params['N_in']
        N_rec   = self.N_rec      = params['N_rec']
        N_out   = self.N_out      = params['N_out']
        N_steps = self.N_steps    = params['N_steps']
        N_batch = self.N_batch = params['N_batch']

        # Physical parameters
        self.dt = params['dt']
        self.tau = params['tau']   #time constant for rec units
        self.alpha = self.dt / self.tau
        self.rec_noise  = params['rec_noise']
        
        # Wiring Min parameters
        self.phi = params.get('phi',.2)  #Rec to Adapt weight
        # 2D distance mat construction
        self.location = np.array(np.meshgrid(range(n_grid),range(n_grid))).reshape([2,n_grid**2]).T
        self.dist_mat = 1.+squareform(pdist(location))
        

        # load weights path
        self.load_weights_path = params.get('load_weights_path', None)

        # regularization coefficients
        self.L1_in = params.get('L1_in', 0)
        self.L1_rec = params.get('L1_rec', 0)
        self.L1_out = params.get('L1_out', 0)

        self.L2_in = params.get('L2_in', 0)
        self.L2_rec = params.get('L2_rec',0)
        self.L2_out = params.get('L2_out',0)

        self.L2_firing_rate = params.get('L2_firing_rate', 0)
        self.sussillo_constant = params.get('sussillo_constant', 0)

        # trainable features
        self.W_in_train = params.get('W_in_train', True)
        self.W_rec_train = params.get('W_rec_train', True)
        self.W_out_train = params.get('W_out_train', True)
        self.b_rec_train = params.get('b_rec_train', True)
        self.b_out_train = params.get('b_out_train', True)
        self.init_state_train = params.get('init_state_train', True)

        # Tensorflow initializations
        self.x = tf.placeholder("float", [N_batch, N_steps, N_in])
        self.y = tf.placeholder("float", [N_batch, N_steps, N_out])
        self.output_mask = tf.placeholder("float", [N_batch, N_steps, N_out])

        # trainable variables
        with tf.variable_scope("model"):

            # ------------------------------------------------
            # Random initialization Load weights from weights path
            # for Initial state, Weight matrices, and bias weights
            # ------------------------------------------------
            if self.load_weights_path is None:
                # random initializations
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


            self.init_state = tf.get_variable('init_state', [N_batch, N_rec],
                                              initializer=init_state_initializer)

            # ------------------------------------------------
            # Trainable variables:
            # Weight matrices and bias weights
            # ------------------------------------------------

            # Input weight matrix:
            # (uniform initialization as in pycog)
            self.W_in = \
                tf.get_variable('W_in', [N_rec, N_in],
                                initializer=W_in_initializer,
                                trainable=self.W_in_train)
            # Recurrent weight matrix:
            # (gamma (Dale) or normal (non-Dale) initialization)
            self.W_rec = \
                tf.get_variable(
                    'W_rec',
                    [N_rec, N_rec],
                    initializer=W_rec_initializer,
                    trainable=self.W_rec_train)
            # Output weight matrix:
            # (uniform initialization as in pycog)
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
            # Network loss
            # ------------------------------------------------
            self.predictions, self.states, self.adapts = self.compute_predictions()
            self.error = self.mean_square_error()
            self.loss = self.error #+ self.regularization()

    # regularized loss function
    def reg_loss(self):
        return self.mean_square_error() + self.regularization() + self.dist_regularizer()

    # mean squared error
    def mean_square_error(self):
        return tf.reduce_mean(self.output_mask*tf.square(self.predictions - self.y))
    
    def dist_regularizer(self):
        return self.phi*tf.reduce_mean(self.W_rec*self.dist_mat)

    # regularizations
    def regularization(self):
        reg = 0

        # L1 weight regularization
        reg += self.L1_in * tf.reduce_mean(tf.abs(self.W_in) * self.input_Connectivity)
        reg += self.L1_rec * tf.reduce_mean(tf.abs(self.W_rec) * self.rec_Connectivity)
        if self.dale_ratio:
            reg += self.L1_out * tf.reduce_mean(tf.matmul(tf.abs(self.W_out) * self.output_Connectivity, self.Dale_out))
        else:
            reg += self.L1_out * tf.reduce_mean(tf.abs(self.W_out) * self.output_Connectivity)

        # L2 weight regularization
        reg += self.L2_in * tf.reduce_mean(tf.square(tf.abs(self.W_in) * self.input_Connectivity))
        reg += self.L2_rec * tf.reduce_mean(tf.square(tf.abs(self.W_rec) * self.rec_Connectivity))
        if self.dale_ratio:
            reg += self.L2_out * tf.reduce_mean(tf.square(
                tf.matmul(tf.abs(self.W_out) * self.output_Connectivity, self.Dale_out)))
        else:
            reg += self.L2_out * tf.reduce_mean(tf.square(tf.abs(self.W_out) * self.output_Connectivity))

        # L2 firing rate regularization
        reg += self.L2_firing_rate * tf.reduce_mean(tf.square(tf.nn.relu(self.states)))

        return reg

    # implement one step of the RNN
    def rnn_step(self, rnn_in, state, adapt):

        #update state
        new_state = ((1-self.alpha) * state)\
                        + self.alpha * (tf.matmul(tf.nn.relu(state),self.W_rec,transpose_b=True, name="1")\
                        + tf.matmul(rnn_in,self.W_in,transpose_b=True, name="2")\
                        + self.b_rec)\
                        - self.adaptation_weight * adapt\
                        + np.sqrt(2.0 * self.alpha * self.rec_noise * self.rec_noise)\
                          * tf.random_normal(state.get_shape(), mean=0.0, stddev=1.0)
          
        #update adaptation parameters
        new_adapt = (1-self.gamma) * adapt + self.gamma * (self.phi * tf.nn.relu(new_state))

        return new_state,new_adapt

    def rnn_output(self, new_state):

        new_output = tf.matmul(tf.nn.relu(new_state),
                            self.W_out, transpose_b=True, name="3") \
                    + self.b_out

        return new_output


    def compute_predictions(self):

        rnn_inputs = tf.unstack(self.x, axis=1)
        state = self.init_state
        adapt = tf.zeros(self.N_rec)
        rnn_outputs = []
        rnn_states = []
        rnn_adapts = []
        for rnn_input in rnn_inputs:
            state,adapt = self.rnn_step(rnn_input, state,adapt)
            output = self.rnn_output(state)
            rnn_outputs.append(output)
            rnn_states.append(state)
            rnn_adapts.append(adapt)
        return tf.transpose(rnn_outputs, [1, 0, 2]), rnn_states, rnn_adapts


    # fix spectral radius of recurrent matrix
    def initial_W(self):
        
        W = np.random.normal(scale=1, size=(self.N_rec, self.N_rec))    
        rho = max(abs(np.linalg.eigvals(W)))
        
        return (1.1/rho) * W 
    

    # train the model using Adam
    def train(self, sess, generator,
              learning_rate=.001, training_iters=50000,
              batch_size=64, display_step=10,weight_save_step=100, save_weights_path= None,
              generator_function= None, training_weights_path = None):


        # train with gradient clipping
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads = optimizer.compute_gradients(self.loss)
        clipped_grads = [(tf.clip_by_norm(grad, 1.0), var)
                         if grad is not None else (grad, var)
                        for grad, var in grads]

        # add vanishing gradient regularizer
        #out, test = self.dOmega_dWrec()
        #clipped_grads[0] = (tf.add(out[0], clipped_grads[0][0]), clipped_grads[0][1])
        #clipped_grads[0] = (tf.Print(clipped_grads[0][0], [clipped_grads[0][0]], "gw_rec"), clipped_grads[0][1])

        optimize = optimizer.apply_gradients(clipped_grads)

        # run session
        sess.run(tf.global_variables_initializer())
        step = 1

        # time training
        t1 = time()
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y, output_mask = generator.next()
            sess.run(optimize, feed_dict={self.x: batch_x, self.y: batch_y, self.output_mask: output_mask})
            if step % display_step == 0:
                # Calculate batch loss
                loss = sess.run(self.loss,
                                feed_dict={self.x: batch_x, self.y: batch_y, self.output_mask: output_mask})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss))

            step += 1
        t2 = time()
        print("Optimization Finished!")
        adapts,states = sess.run([self.adapts,self.states], feed_dict={self.x: batch_x})

        # save weights
        if save_weights_path is not None:
            np.savez(save_weights_path, W_in = self.W_in.eval(session=sess),
                                    W_rec = self.W_rec.eval(session=sess),
                                    W_out = self.W_out.eval(session=sess),
                                    b_rec = self.b_rec.eval(session=sess),
                                    b_out = self.b_out.eval(session=sess),
                                    init_state = self.init_state.eval(session=sess))

        return (t2 - t1), adapts, states, [batch_x, batch_y]


    # use a trained model to get test outputs
    def test(self, sess, rnn_in, weights_path = None):
        if(weights_path):
            saver = tf.train.Saver()
            # Restore variables from disk.
            saver.restore(sess, weights_path)
            predictions, states = sess.run([self.predictions, self.states], feed_dict={self.x: rnn_in})
        else:
            predictions, states = sess.run([self.predictions, self.states], feed_dict={self.x: rnn_in})

        return predictions, states