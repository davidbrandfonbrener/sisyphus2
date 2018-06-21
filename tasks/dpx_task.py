
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'viridis'

import numpy as np
import tensorflow as tf
from backend.networks import Model
from backend.weight_initializer import weight_initializer
#import backend.visualizations as V
from backend.simulation_tools import Simulator
from scipy.spatial.distance import pdist, squareform



# Builds a dictionary of parameters that specifies the information
# about an instance of this specific task
def set_params(trial_proportion='prepotent',n_in = 4, n_out = 2, input_wait = 30, mem_gap = 40, stim_dur = 10,
               out_gap = 0, out_dur=20, var_delay_length = 0, var_in_wait=0,
               var_out_gap = 0, stim_noise = 0, rec_noise = .1, L1_rec = 0, L1_in = 0, L1_out = 0, 
               L2_firing_rate = 1, sample_size = 128, epochs = 100,
               N_rec = 50, dale_ratio=0.8, tau=100.0, dt = 10.0, biases = True,
               second_in_scale = 1, go_cue= True,task='xor',init_type= 'gauss'):
    
    params = dict()
    params['trial_proportion'] = trial_proportion
    params['go_cue']           = go_cue
    params['N_in']             = n_in
    if go_cue:
        params['N_in']         = n_in+1
    params['N_out']            = n_out
    params['N_steps']          = input_wait + var_in_wait + stim_dur + mem_gap + var_delay_length + stim_dur + out_gap + var_out_gap + out_dur
    params['N_batch']          = sample_size
    params['init_type']        = init_type
    params['input_wait']       = input_wait
    params['mem_gap']          = mem_gap
    params['stim_dur']         = stim_dur
    params['out_gap']          = out_gap
    params['out_dur']          = out_dur
    params['var_delay_length'] = var_delay_length
    params['var_in_wait']      = var_in_wait
    params['var_out_gap']      = var_out_gap
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
    params['L1_in']            = L1_in
    params['L1_out']           = L1_out
    params['L2_firing_rate']   = L2_firing_rate
    params['biases']           = biases
    params['second_in_scale']  = second_in_scale #If = 0, no second input

    return params

# This generates the training data for our network
# It will be a set of input_times and output_times for when we expect input
# and when the corresponding output is expected
def build_train_trials(params):
    n_in = params['N_in']
    n_out = params['N_out']
    input_wait = params['input_wait']
    mem_gap = params['mem_gap']
    stim_dur = params['stim_dur']
    out_gap = params['out_gap']
    out_dur = params['out_dur']
    var_delay_length = params['var_delay_length']
    var_in_wait = params['var_in_wait']
    var_out_gap = params['var_out_gap']
    stim_noise = params['stim_noise']
    sample_size = int(params['sample_size'])
    task = params['task']
    second_in_scale = params['second_in_scale']
    go_cue = params['go_cue']
    trial_proportion = params['trial_proportion']

    #set up variable timing(if applicable)
    if var_delay_length == 0:
        var_delay = np.zeros(sample_size, dtype=int)
    else:
        var_delay = np.random.randint(var_delay_length, size=sample_size) + 1

    if var_in_wait == 0:
        var_in = np.zeros(sample_size, dtype=int)
    else:
        var_in = np.random.randint(var_in_wait, size=sample_size) + 1

    if var_out_gap == 0:
        var_out = np.zeros(sample_size, dtype=int)
    else:
        var_out = np.random.randint(var_out_gap, size=sample_size) + 1

    seq_dur = input_wait + var_in_wait + stim_dur + mem_gap + var_delay_length + stim_dur + out_gap + var_out_gap + out_dur


    input_times = np.zeros([sample_size, n_in], dtype=np.int)
    output_times = np.zeros([sample_size, 1], dtype=np.int)

    #initialize
    x_train = np.zeros([sample_size, seq_dur, n_in])
    y_train = 0.1 * np.ones([sample_size, seq_dur, n_out])
    mask = np.ones((sample_size, seq_dur, n_out))
    
    #trial types
    trial_types = [[0,0],[0,1],[1,0],[1,1]]
    
    for sample in np.arange(sample_size):
        
        #modulate between balanced or prepotent conditions
        if trial_proportion == 'balanced':
            trial_idx = np.random.choice(range(4),p=[.25,.25,.25,.25])
            input_pattern = trial_types[trial_idx]
        elif trial_proportion == 'prepotent':
            trial_idx = np.random.choice(range(4),p=[.69,.125,.125,.06])
            input_pattern = trial_types[trial_idx]

        in_period1 = range(input_wait+var_in[sample],(input_wait+var_in[sample]+stim_dur))
        in_period2 = range(input_wait+var_in[sample]+stim_dur+mem_gap+var_delay[sample],
                           (input_wait+var_in[sample]+stim_dur+mem_gap+var_delay[sample]+stim_dur))
        
        out_period = range(input_wait+var_in[sample]+stim_dur+mem_gap+var_delay[sample]+ stim_dur + out_gap + var_out[sample],
                           input_wait+var_in[sample]+stim_dur+mem_gap+var_delay[sample]+stim_dur+ out_gap + var_out[sample] + out_dur)
        
        go_cue_period = range(input_wait+var_in[sample]+stim_dur+mem_gap+var_delay[sample]+ stim_dur + out_gap + var_out[sample] - stim_dur,
                           input_wait+var_in[sample]+stim_dur+mem_gap+var_delay[sample]+stim_dur+ out_gap + var_out[sample])

        #generic stimuli
        x_train[sample,in_period1,input_pattern[0]] = 1
        x_train[sample,in_period2,input_pattern[1]+2] = 1 * second_in_scale #input_pattern[sample,input_order[sample,1]]

        
        #go cue input
        if go_cue:
            x_train[sample,go_cue_period,-1] = 1     #set up go cue

        #conditonal output
        if trial_idx == 0:
            y_train[sample,out_period,0] = 1
        else:
            y_train[sample,out_period,1] = 1
        

        
        #Mask output after response epoch
        mask[sample,range(input_wait+var_in[sample]+stim_dur+mem_gap+var_delay[sample]+stim_dur+out_gap+out_dur,seq_dur),:] = 0
        #Mask output during early response epoch
        mask[sample,range(input_wait+var_in[sample]+stim_dur+mem_gap+var_delay[sample]+ stim_dur + out_gap + var_out[sample],
                          input_wait+var_in[sample]+stim_dur+mem_gap+var_delay[sample]+ stim_dur + out_gap + var_out[sample]+10),:] = 0
        #Mask until output
        mask[sample,range(0,input_wait+var_in[sample]+stim_dur+mem_gap+var_delay[sample]+ stim_dur + out_gap + var_out[sample]+10),:] = 0
             
    #note:#TODO im doing a quick fix, only considering 1 ouput neuron
    
    #for sample in np.arange(sample_size):
    #    mask[sample, :, 0] = [0.0 if x == .5 else 1.0 for x in y_train[sample, :, :]]
    #mask = np.array(mask, dtype=float)

    x_train = x_train + stim_noise * np.random.randn(sample_size, seq_dur, n_in)
    params['input_times'] = input_times
    params['output_times'] = output_times
    return x_train, y_train, mask

def generate_train_trials(params):
    while 1 > 0:
        yield build_train_trials(params)
        
        
            
if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', help="task name", type=str)
    parser.add_argument('fig_directory',help="where to save figures")
    parser.add_argument('weights_path',help="where to save weights")
    parser.add_argument('-m','--mem_gap', help="supply memory gap length", type=int,default=50)
    parser.add_argument('-v','--var_delay', help="supply variable memory gap delay", type=int,default=0)
    parser.add_argument('-r','--rec_noise', help ="recurrent noise", type=float,default=0.0)
    parser.add_argument('-i','--initialization', help ="initialization of Wrec", type=str,default='gauss')
    parser.add_argument('-t','--training_iters', help="training iterations", type=int,default=300000)
    parser.add_argument('-ts','--task',help="task type",default='memory_saccade')
    args = parser.parse_args()
    
    #run params
    run_name = args.run_name
    fig_directory = args.fig_directory
    
    #initialization of wrec
    init_type = args.initialization
    
    #task params
    mem_gap_length = args.mem_gap
    input_wait = 40
    stim_dur = 10
    out_gap = 0
    out_dur = 60
    
    var_delay_length = args.var_delay
    var_in_wait = 40
    var_out_gap = 0
    second_in_scale = 0.  #Only one input period or two (e.g. mem saccade no distractor vs with distractor)
    task = args.task
    
    #model params
    n_in = 2 
    n_hidden = 100 
    n_out = 2
    #n_steps = 80 
    tau = 100.0 #As double
    dt = 20.0  #As double
    dale_ratio = 0
    rec_noise = args.rec_noise
    stim_noise = 0.1
    batch_size = 128
    
    
    #train params
    learning_rate = .0001 
    training_iters = args.training_iters
    display_step = 200
    
    #weights_path = '../weights/' + run_name + '.npz'
    
    params = set_params(epochs=200, sample_size= batch_size, input_wait=input_wait, 
                        stim_dur=stim_dur, mem_gap=mem_gap_length, out_gap = out_gap, out_dur=out_dur, 
                        N_rec=n_hidden, n_out = n_out, n_in = n_in, 
                        var_delay_length=var_delay_length,
                        var_in_wait = var_in_wait, var_out_gap = var_out_gap,
                        rec_noise=rec_noise, stim_noise=stim_noise, 
                        dale_ratio=dale_ratio, tau=tau, task=task,
                        second_in_scale=second_in_scale,init_type=init_type)
    
    
    output_weights_path = args.weights_path
    
    'external weight intializer class'
    autapses = True
    w_initializer = weight_initializer(params,output_weights_path[:-4] + '_init',autapses=autapses)
    input_weights_path = w_initializer.gen_weight_dict()
    params['load_weights_path'] = input_weights_path + '.npz'
    
    generator = generate_train_trials(params)
    #model = Model(n_in, n_hidden, n_out, n_steps, tau, dt, dale_ratio, rec_noise, batch_size)
    model = Model(params)
    sess = tf.Session()
    
    
    
    model.train(sess, generator, learning_rate = learning_rate, 
                training_iters = training_iters, save_weights_path = output_weights_path)
    
    
    sess.close()



