

import numpy as np
import tensorflow as tf
from backend.networks import Model
#import backend.visualizations as V
from backend.simulation import Simulator
#import matplotlib.pyplot as plt


# Builds a dictionary of parameters that specifies the information
# about an instance of this specific task
def set_params(n_in = 1, n_out = 1, n_back = 0, n_steps = 200, stim_noise = 0, rec_noise = 0, L1_rec = 0, L2_firing_rate = 0,
                    sample_size = 128, epochs = 100, N_rec = 50, dale_ratio=0.8, tau=100.0, dt = 10.0, biases = False, task='n_back'):
    params = dict()
    params['N_in']             = n_in
    params['N_out']            = n_out
    params['N_steps']          = n_steps
    params['N_batch']          = sample_size
    params['stim_noise']       = stim_noise
    params['rec_noise']        = rec_noise
    params['sample_size']      = sample_size
    params['epochs']           = epochs
    params['N_rec']            = N_rec
    params['dale_ratio']       = dale_ratio
    params['tau']              = tau
    params['dt']               = dt
    params['alpha']            = dt/tau
    params['task']             = task
    params['L1_rec']           = L1_rec
    params['L2_firing_rate']   = L2_firing_rate
    params['N_back']           = n_back
    params['biases']           = biases

    return params

# This generates the training data for our network
# It will be a set of input_times and output_times for when we expect input
# and when the corresponding output is expected
def build_train_trials(params):
    n_in = params['N_in']
    n_steps = params['N_steps']
    #input_wait = params['input_wait']
    #mem_gap = params['mem_gap']
    #stim_dur = params['stim_dur']
    #out_dur = params['out_dur']
    #var_delay_length = params['var_delay_length']
    n_back = params['N_back']
    stim_noise = params['stim_noise']
    batch_size = int(params['sample_size'])
    #task = params['task']

    input_times = np.zeros([batch_size, n_in], dtype=np.int)
    output_times = np.zeros([batch_size, n_out], dtype=np.int)

    x_train = np.zeros([batch_size,n_steps,n_in])
    y_train = np.zeros([batch_size,n_steps,n_out])
    mask = np.ones((batch_size, n_steps, n_in))
    
    stim_times = range(0,n_steps,40)
    stim_dir = np.random.choice([-1,1],replace=True,size=(len(stim_times),batch_size))
    for ii in range(batch_size):
        for jj in range(len(stim_times)-n_back):
            x_train[ii,range(stim_times[jj],stim_times[jj]+10),0] = stim_dir[jj,ii]
            y_train[ii,range(stim_times[jj]+15+40*n_back,stim_times[jj]+25+40*n_back),0] = stim_dir[jj,ii]

    #note:#TODO im doing a quick fix, only considering 1 ouput neuron
    
    #for sample in np.arange(sample_size):
    #    mask[sample, :, 0] = [0.0 if x == .5 else 1.0 for x in y_train[sample, :, :]]
    #mask = np.array(mask, dtype=float)

    x_train = x_train + stim_noise * np.random.randn(batch_size, n_steps, n_in)
    params['input_times'] = input_times
    params['output_times'] = output_times
    return x_train, y_train, mask

def generate_train_trials(params):
    while 1 > 0:
        yield build_train_trials(params)

        
if __name__ == "__main__":
    
#    import argparse
#    
#    parser = argparse.ArgumentParser()
#    parser.add_argument('mem_gap', help="supply memory gap length", type=int)
#    args = parser.parse_args()
#    
#    mem_gap_length = args.mem_gap
    
    #model params
    n_in = 1
    n_hidden = 50 
    n_out = 1
    #n_steps = 80 
    tau = 100.0 #As double
    dt = 20.0  #As double
    dale_ratio = 0
    rec_noise = 0.0
    stim_noise = 0.0
    batch_size = 128
    #var_delay_length = 50
    
    n_back = 1
    
    #train params
    learning_rate = .0001 
    training_iters = 300000
    display_step = 200
    
    weights_path = '../weights/n_back.npz'
    #weights_path = None
    
    params = set_params(n_in = n_in, n_out = n_out, n_back = n_back, n_steps = 200, stim_noise = stim_noise, rec_noise = rec_noise, L1_rec = 0, L2_firing_rate = 0,
                    sample_size = 128, epochs = 100, N_rec = 50, dale_ratio=dale_ratio, tau=tau, dt = dt, task='n_back')
    generator = generate_train_trials(params)
    #model = Model(n_in, n_hidden, n_out, n_steps, tau, dt, dale_ratio, rec_noise, batch_size)
    model = Model(params)
    sess = tf.Session()
    
    
    
    model.train(sess, generator, learning_rate = learning_rate, training_iters = training_iters, weights_path = weights_path)

    data = generator.next()
    #output,states = model.test(sess, input, weights_path = weights_path)
    
    
    W = model.W_rec.eval(session=sess)
    U = model.W_in.eval(session=sess)
    Z = model.W_out.eval(session=sess)
    brec = model.b_rec.eval(session=sess)
    bout = model.b_out.eval(session=sess)
    
    sim = Simulator(params, weights_path=weights_path)
    output,states = sim.run_trial(data[0][0,:,:],t_connectivity=False)
    
    sess.close()
