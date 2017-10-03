import numpy as np
import tensorflow as tf
from backend.networks import Model
import backend.visualizations as V
from backend.simulation_tools import Simulator
import matplotlib.pyplot as plt


# Builds a dictionary of parameters that specifies the information
# about an instance of this specific task
def set_params(n_in = 2, n_out = 2, input_wait = 3, mem_gap = 4, stim_dur = 3, out_dur=5,
                    var_delay_length = 0, stim_noise = 0, rec_noise = .1, L1_rec = 0, L2_firing_rate = 1,
                    sample_size = 128, epochs = 100, N_rec = 50, dale_ratio=0.8, tau=100.0, dt = 10.0, task='xor'):
    params = dict()
    params['N_in']             = n_in
    params['N_out']            = n_out
    params['N_steps']          = input_wait + stim_dur + mem_gap + var_delay_length +stim_dur + out_dur
    params['N_batch']          = sample_size
    params['input_wait']       = input_wait
    params['mem_gap']          = mem_gap
    params['stim_dur']         = stim_dur
    params['out_dur']          = out_dur
    params['var_delay_length'] = var_delay_length
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

    return params

# This generates the training data for our network
# It will be a set of input_times and output_times for when we expect input
# and when the corresponding output is expected
def build_train_trials(params):
    n_in = params['N_in']
    input_wait = params['input_wait']
    mem_gap = params['mem_gap']
    stim_dur = params['stim_dur']
    out_dur = params['out_dur']
    var_delay_length = params['var_delay_length']
    stim_noise = params['stim_noise']
    sample_size = int(params['sample_size'])
    task = params['task']

    if var_delay_length == 0:
        var_delay = np.zeros(sample_size, dtype=int)
    else:
        var_delay = np.random.randint(var_delay_length, size=sample_size) + 1

    seq_dur = input_wait + stim_dur + mem_gap + var_delay_length + stim_dur + out_dur

    input_pattern = np.random.randint(2,size=(sample_size,2))
    #input_order = np.random.randint(2,size=(sample_size,2))
    if task == 'xor':
        output_pattern = (np.sum(input_pattern,1) == 1).astype('float') #xor
    elif task == 'or':
        output_pattern = (np.sum(input_pattern,1) >= 1).astype('float') #or
    elif task == 'and':
        output_pattern = (np.sum(input_pattern,1) >= 2).astype('float') #and
    elif task == 'memory_saccade':
        output_pattern = input_pattern[:,0] #input_pattern[range(np.shape(input_pattern)[0]),input_order[:,0]]                             #memory saccade with distractor

    input_times = np.zeros([sample_size, n_in], dtype=np.int)
    output_times = np.zeros([sample_size, 1], dtype=np.int)


    x_train = np.zeros([sample_size, seq_dur, 2])
    y_train = 0.1 * np.ones([sample_size, seq_dur, 2])
    mask = np.ones((sample_size, seq_dur, 2))
    for sample in np.arange(sample_size):

        in_period1 = range(input_wait,(input_wait+stim_dur))
        in_period2 = range(input_wait+stim_dur+mem_gap+var_delay[sample],(input_wait+stim_dur+mem_gap+var_delay[sample]+stim_dur))
        x_train[sample,in_period1,input_pattern[sample,0]] = 1
        x_train[sample,in_period2,input_pattern[sample,1]] = 1 #input_pattern[sample,input_order[sample,1]]
        
        out_period = range(input_wait+stim_dur+mem_gap+var_delay[sample]+stim_dur,input_wait+stim_dur+mem_gap+var_delay[sample]+stim_dur+out_dur)
        y_train[sample,out_period,output_pattern[sample]] = 1
        mask[sample,range(input_wait+stim_dur+mem_gap+var_delay[sample]+stim_dur+out_dur,seq_dur),:] = 0

    #note:#TODO im doing a quick fix, only considering 1 ouput neuron
    
    #for sample in np.arange(sample_size):
    #    mask[sample, :, 0] = [0.0 if x == .5 else 1.0 for x in y_train[sample, :, :]]
    #mask = np.array(mask, dtype=float)

    x_train = x_train + stim_noise * np.random.randn(sample_size, seq_dur, 2)
    params['input_times'] = input_times
    params['output_times'] = output_times
    return x_train, y_train, mask

def generate_train_trials(params):
    while 1 > 0:
        yield build_train_trials(params)

        
if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('mem_gap', help="supply memory gap length", type=int)
    args = parser.parse_args()
    
    mem_gap_length = args.mem_gap
    
    #model params
    n_in = 2 
    n_hidden = 100 
    n_out = 2
    #n_steps = 80 
    tau = 100.0 #As double
    dt = 20.0  #As double
    dale_ratio = 0.8
    rec_noise = 0.0
    stim_noise = 0.0
    batch_size = 128
    var_delay_length = 50
    
    #train params
    learning_rate = .001 
    training_iters = 30000
    display_step = 200
    
    weights_path = '../weights/mem_sac_' + str(mem_gap_length) + '.npz'
    #weights_path = None
    
    params = set_params(epochs=200, sample_size= batch_size, input_wait=10, stim_dur=10, mem_gap=mem_gap_length, out_dur=30, N_rec=n_hidden,
                        n_out = n_out, n_in = n_in, var_delay_length=var_delay_length,
                        rec_noise=rec_noise, stim_noise=stim_noise, dale_ratio=dale_ratio, tau=tau, task='memory_saccade')
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
    output,states = sim.run_trials(data[0], 50)
    
    sess.close()



