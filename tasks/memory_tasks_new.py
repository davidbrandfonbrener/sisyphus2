import numpy as np
import tensorflow as tf
from backend.networks import Model
import backend.visualizations as V
from backend.simulation_tools import Simulator
import matplotlib.pyplot as plt


# Builds a dictionary of parameters that specifies the information
# about an instance of this specific task
def set_params(n_in = 2, n_out = 2, input_wait = 3, mem_gap = 4, stim_dur = 3, 
               out_gap = 0, out_dur=5, var_delay_length = 0, var_in_wait=0,
               var_out_gap = 0, stim_noise = 0, rec_noise = .1, L1_rec = 0, 
               L2_firing_rate = 1, sample_size = 128, epochs = 100,
               N_rec = 50, dale_ratio=0.8, tau=100.0, dt = 10.0, biases = True,
               second_in_scale = 1, go_cue= True,task='xor'):
    
    params = dict()
    params['go_cue']           = go_cue
    params['N_in']             = n_in
    if go_cue:
        params['N_in']         = n_in+1
    params['N_out']            = n_out
    params['N_steps']          = input_wait + var_in_wait + stim_dur + mem_gap + var_delay_length + stim_dur + out_gap + var_out_gap + out_dur
    params['N_batch']          = sample_size
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

    input_pattern = np.random.randint(2,size=(sample_size,2))
    #input_order = np.random.randint(2,size=(sample_size,2))
    if task == 'xor':
        output_pattern = (np.sum(input_pattern,1) == 1).astype('int') #xor
    elif task == 'or':
        output_pattern = (np.sum(input_pattern,1) >= 1).astype('int') #or
    elif task == 'and':
        output_pattern = (np.sum(input_pattern,1) >= 2).astype('int') #and
    elif task == 'memory_saccade':
        output_pattern = input_pattern[:,0] #input_pattern[range(np.shape(input_pattern)[0]),input_order[:,0]]                             #memory saccade with distractor
        

    input_times = np.zeros([sample_size, n_in], dtype=np.int)
    output_times = np.zeros([sample_size, 1], dtype=np.int)


    x_train = np.zeros([sample_size, seq_dur, n_in])
    y_train = 0.1 * np.ones([sample_size, seq_dur, n_out])
    mask = np.ones((sample_size, seq_dur, n_out))
    for sample in np.arange(sample_size):

        in_period1 = range(input_wait+var_in[sample],(input_wait+var_in[sample]+stim_dur))
        in_period2 = range(input_wait+var_in[sample]+stim_dur+mem_gap+var_delay[sample],
                           (input_wait+var_in[sample]+stim_dur+mem_gap+var_delay[sample]+stim_dur))
        
        out_period = range(input_wait+var_in[sample]+stim_dur+mem_gap+var_delay[sample]+ stim_dur + out_gap + var_out[sample],
                           input_wait+var_in[sample]+stim_dur+mem_gap+var_delay[sample]+stim_dur+ out_gap + var_out[sample] + out_dur)
        
        go_cue_period = range(input_wait+var_in[sample]+stim_dur+mem_gap+var_delay[sample]+ stim_dur + out_gap + var_out[sample] - stim_dur,
                           input_wait+var_in[sample]+stim_dur+mem_gap+var_delay[sample]+stim_dur+ out_gap + var_out[sample])
        
        x_train[sample,in_period1,input_pattern[sample,0]] = 1
        x_train[sample,in_period2,input_pattern[sample,1]] = 1 * second_in_scale #input_pattern[sample,input_order[sample,1]]
        if go_cue:
            x_train[sample,go_cue_period,2] = 1     #set up go cue   
        

        y_train[sample,out_period,output_pattern[sample]] = 1
        
        mask[sample,range(input_wait+var_in[sample]+stim_dur+mem_gap+var_delay[sample]+stim_dur+out_gap+out_dur,seq_dur),:] = 0

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
        
def state_tensor_decomposition(states,rank=3):
    import tensorly as tly
    from tensorly.decomposition import parafac
    
    s_tens = tly.tensor(states)
    f = parafac(s_tens,rank=rank)
    factors = []
    
    for ii in range(len(f)):
        factors.append(tly.to_numpy(f[ii]))
    
    return factors
    
def fixed_point_analysis(states,W,brec):
    
    n_steps = states.shape[0]
    n_hidden = states.shape[1]

    Weff = np.zeros([n_hidden,n_hidden,n_steps])
    
    fps = []

    for ii in range(n_steps):
        partition = states[ii,:]>0
        Weff[:,:,ii] = W*partition
        fp_putative = np.linalg.inv(np.eye(n_hidden)-Weff[:,:,ii]).dot(brec)
        in_partition = np.array_equal( fp_putative>0, partition)  
        
        stable = np.max(np.linalg.eig(Weff[:,:,ii])[0].real)<1

        if in_partition:
            fps.append(dict(fp=fp_putative,partition=partition,stability=stable,time=ii))
        
    
    return fps
    
def long_delay_test(sim):
    d0 = np.zeros([3000,3])
    d1 = np.zeros([3000,3])
    dflip = np.zeros([5000,3])
    
    for ii in range(10,5000,1000):
        dflip[ii:ii+10,np.random.randint(low=0,high=2)] = 1
    
    d0[50:60,0] = 1
    d1[50:60,1] = 1

    o0,s0 = sim.run_trial(d0,t_connectivity=False)
    o1,s1 = sim.run_trial(d1,t_connectivity=False)
    oflip,sflip = sim.run_trial(dflip,t_connectivity=False)
    
    plt.figure(figsize=(6,12))
    plt.subplot(2,1,1)
    plt.plot(sflip[:,0,:])
    plt.subplot(8,1,5)
    plt.plot(dflip)
    plt.subplot(8,1,6)
    plt.plot(oflip[:,0,:])
    
    plt.show()
    
    return o0,s0,o1,s1
    
def plot_by_max(state,norm=True,thresh=.001):
    fr = np.maximum(state,thresh)
    if norm:
        #fr = ((fr-np.mean(fr,axis=0))/np.std(fr,axis=0))
        fr = fr/np.max(fr,axis=0)
    idx = np.argsort(np.argmax(fr,axis=0))
    plt.pcolormesh(fr[:,idx].T)
    plt.colorbar()
    plt.xlim([0,np.shape(fr)[0]])
    
def plot_dist_to_fixed(state,fp):
    d = np.zeros(np.shape(state)[0])
    for ii in range(np.shape(state)[0]):
        d[ii] = np.sum((fp-state[ii,:])**2)
    plt.plot(d,'.')
    plt.ylim([0,np.max(d)*1.5])
    return d
    

        
if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('task_name', help="task name", type=str)
    parser.add_argument('-m','--mem_gap', help="supply memory gap length", type=int,default=50)
    parser.add_argument('-v','--var_delay', help="supply variable memory gap delay", type=int,default=0)
    parser.add_argument('-r','--rec_noise', help ="recurrent noise", type=float,default=0.0)
    parser.add_argument('-t','--training_iters', help="training iterations", type=int,default=300000)
    args = parser.parse_args()
    
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
    task = 'memory_saccade'
    name = args.task_name
    
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
    
    weights_path = '../weights/' + name + '_' + str(mem_gap_length) + '.npz'
    #weights_path = None
    
    params = set_params(epochs=200, sample_size= batch_size, input_wait=input_wait, 
                        stim_dur=stim_dur, mem_gap=mem_gap_length, out_gap = out_gap, out_dur=out_dur, 
                        N_rec=n_hidden, n_out = n_out, n_in = n_in, 
                        var_delay_length=var_delay_length,
                        var_in_wait = var_in_wait, var_out_gap = var_out_gap,
                        rec_noise=rec_noise, stim_noise=stim_noise, 
                        dale_ratio=dale_ratio, tau=tau, task=task,
                        second_in_scale=second_in_scale)
    
    generator = generate_train_trials(params)
    #model = Model(n_in, n_hidden, n_out, n_steps, tau, dt, dale_ratio, rec_noise, batch_size)
    model = Model(params)
    sess = tf.Session()
    
    
    
    model.train(sess, generator, learning_rate = learning_rate, 
                training_iters = training_iters, weights_path = weights_path)

    data = generator.next()
    #output,states = model.test(sess, input, weights_path = weights_path)
    
    
    W = model.W_rec.eval(session=sess)
    U = model.W_in.eval(session=sess)
    Z = model.W_out.eval(session=sess)
    brec = model.b_rec.eval(session=sess)
    bout = model.b_out.eval(session=sess)
    
    sim = Simulator(params, weights_path=weights_path)
    output,states = sim.run_trial(data[0][0,:,:],t_connectivity=False)
    
    s = np.zeros([states.shape[0],batch_size,n_hidden])
    for ii in range(batch_size): 
        s[:,ii,:] = sim.run_trial(data[0][ii,:,:],t_connectivity=False)[1].reshape([states.shape[0],n_hidden])
    
    sess.close()



