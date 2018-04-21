import numpy as np
import tensorflow as tf

class WeightInitializer(object):

    def __init__(self, **kwargs):

        self.initializations = dict()

        return


    def get(self, tensor_name):

        return tf.constant_initializer(self.initializations[tensor_name])


    def save(self, save_path):

        np.savez(save_path, **self.initializations)
        return







class GaussianSpectralRadius(WeightInitializer):
    '''Generate random gaussian weights with specified spectral radius'''

    def __init__(self, N_in, N_rec, N_out, spec_rad, autapses):
        self.initializations = dict()

        # Uniform between -.1 and .1
        self.initializations['W_in'] = .2 * np.random.rand(N_rec, N_in) - .1
        self.initializations['W_out'] = .2 * np.random.rand(N_out, N_rec) - .1

        self.initializations['b_rec'] = np.zeros(N_rec)
        self.initializations['b_out'] = np.zeros(N_out)

        self.initializations['init_state'] = .1 + .01 * np.random.randn(N_rec)

        W_rec = np.random.randn(N_rec, N_rec)
        self.initializations['W_rec'] = spec_rad * W_rec / np.max(np.abs(np.linalg.eig(W_rec)[0]))

        self.initializations['input_Connectivity'] = np.ones([N_rec, N_in])
        self.initializations['rec_Connectivity'] = np.ones([N_rec, N_rec])
        self.initializations['output_Connectivity'] = np.ones([N_out, N_rec])

        if not autapses:
            self.initializations['W_rec'][np.eye(N_rec) == 1] = 0
            self.initializations['rec_Connectivity'][np.eye(N_rec) == 1] = 0

        return


class AlphaIdentity(WeightInitializer):
    '''Generate recurrent weights w(i,i) = alpha, w(i,j) = 0'''

    def __init__(self, N_in, N_rec, N_out, alpha, autapses):

        self.initializations = dict()

        # Uniform between -.1 and .1
        self.initializations['W_in'] = .2 * np.random.rand(N_rec, N_in) - .1
        self.initializations['W_out'] = .2 * np.random.rand(N_out, N_rec) - .1

        self.initializations['b_rec'] = np.zeros(N_rec)
        self.initializations['b_out'] = np.zeros(N_out)

        self.initializations['init_state'] = .1 + .01 * np.random.randn(N_rec)

        self.initializations['W_rec'] = np.eye(N_rec) * alpha

        self.initializations['input_Connectivity'] = np.ones([N_rec, N_in])
        self.initializations['rec_Connectivity'] = np.ones([N_rec, N_rec])
        self.initializations['output_Connectivity'] = np.ones([N_out, N_rec])

        if not autapses:
            self.initializations['W_rec'][np.eye(N_rec) == 1] = 0
            self.initializations['rec_Connectivity'][np.eye(N_rec) == 1] = 0

        return


class FeedForward(WeightInitializer):
    '''Generate random feedforward wrec (lower triangular)'''

    def __init__(self, N_in, N_rec, N_out, autapses, sigma):

        self.initializations = dict()

        # Uniform between -.1 and .1
        self.initializations['W_in'] = .2 * np.random.rand(N_rec, N_in) - .1
        self.initializations['W_out'] = .2 * np.random.rand(N_out, N_rec) - .1

        self.initializations['b_rec'] = np.zeros(N_rec)
        self.initializations['b_out'] = np.zeros(N_out)

        self.initializations['init_state'] = .1 + .01 * np.random.randn(N_rec)

        self.initializations['W_rec'] = np.tril(sigma * np.random.randn(N_rec, N_rec), -1)

        self.initializations['input_Connectivity'] = np.ones([N_rec, N_in])
        self.initializations['rec_Connectivity'] = np.ones([N_rec, N_rec])
        self.initializations['output_Connectivity'] = np.ones([N_out, N_rec])

        if not autapses:
            self.initializations['W_rec'][np.eye(N_rec) == 1] = 0
            self.initializations['rec_Connectivity'][np.eye(N_rec) == 1] = 0

        return