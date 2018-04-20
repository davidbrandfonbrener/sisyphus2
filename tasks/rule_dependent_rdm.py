import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'viridis'

import numpy as np
import tensorflow as tf
from backend.networks import Model
#import backend.visualizations as V
from backend.weight_initializer import WeightInitializer
from backend.simulation_tools import Simulator

# Builds a dictionary of parameters that specifies the information
# about an instance of this specific task
def set_params(n_rules = 2, n_out = 2, n_steps = 200, coherences=[.5], stim_noise = 0, rec_noise = 0, L1_rec = 0, L2_firing_rate = 0,
                    sample_size = 128, epochs = 100, N_rec = 50, dale_ratio=0.8, tau=100.0, dt = 10.0, biases = True,
               task='n_back', rt_version=False):
    params = dict()
    params['N_in']             = n_rules*2 
    params['N_out']            = n_out
    params['N_batch']          = sample_size
    params['N_steps']          = n_steps
    params['N_rules']          = n_rules
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
    params['coherences']       = coherences
    params['rt_version']       = rt_version

    return params

# This generates the training data for our network
# It will be a set of input_times and output_times for when we expect input
# and when the corresponding output is expected
def build_train_trials(params):
    n_in = params['N_in']
    n_out = params['N_out']
    n_steps = params['N_steps']
    n_rules = params['N_rules']
    
    stim_noise = params['stim_noise']
    batch_size = int(params['sample_size'])
    rt_version = params['rt_version']
    coherences = params['coherences']

    
    input_times = np.zeros([batch_size, n_in], dtype=np.int)
    output_times = np.zeros([batch_size, n_out], dtype=np.int)

    x_train = np.zeros([batch_size,n_steps,n_in])
    y_train = np.zeros([batch_size,n_steps,n_out]) + .1
    mask = np.ones((batch_size, n_steps, n_out))
    
    stim_time = range(100,180)
    if rt_version:
        out_time = range(50,200)
    else:
        out_time = range(160,200)
        
    rule_display_time = range(40,80)

    rules = np.random.choice(range(n_rules),replace=True,size=(batch_size))
    dirs = np.random.choice([-1,1],replace=True,size=(batch_size,n_rules))
    cohs = np.random.choice(coherences,replace=True,size=(batch_size,n_rules))
    stims = dirs*cohs;
    
    for ii in range(batch_size):
        x_train[ii,rule_display_time,rules[ii]] = 1
        x_train[ii,stim_time,n_rules:] = stims[ii,:].reshape([n_rules,1]).dot(np.ones([1,len(stim_time)])).T
        y_train[ii,out_time,(dirs[ii,rules[ii]]+1)/2] = 1 #one hot out (convert from dir (-1,1) to 0,1)

    x_train = x_train + stim_noise * np.random.randn(batch_size, n_steps, n_in)
    params['input_times'] = input_times
    params['output_times'] = output_times

    #plt.plot(range(len(x_train[0,:,0])), x_train[0,:,0])
    #plt.show()
    #plt.plot(range(len(y_train[0, :, 0])), y_train[0, :, 0])
    #plt.show()

    return x_train, y_train, mask
    

def generate_train_trials(params):
    while 1 > 0:
        yield build_train_trials(params)
        
        
def build_test_trials(params):
    
    n_in = params['N_in']
    n_out = params['N_out']
    n_steps = params['N_steps']
    n_rules = params['N_rules']
    
    stim_noise = params['stim_noise']
    batch_size = int(params['sample_size'])
    rt_version = params['rt_version']
    coherences = params['coherences']

    input_times = np.zeros([batch_size, n_in], dtype=np.int)
    output_times = np.zeros([batch_size, n_out], dtype=np.int)

    x_train = np.zeros([batch_size,n_steps,n_in])
    y_train = np.zeros([batch_size,n_steps,n_out])
    mask = np.ones((batch_size, n_steps, n_out))
    
    stim_time = range(40,140)
    if rt_version:
        out_time = range(50,200)
    else:
        out_time = range(160,200)

    rules = np.random.choice(range(n_rules),replace=True,size=(batch_size))
    dirs = np.random.choice([-1,1],replace=True,size=(batch_size,n_rules))
    cohs = np.random.choice(coherences,replace=True,size=(batch_size,n_rules))
    stims = dirs*cohs;
    
    for ii in range(batch_size):
        x_train[ii,stim_time,:] = stims[ii,:].reshape([n_rules,1]).dot(np.ones([1,len(stim_time)])).T
        y_train[ii,out_time,0] = dirs[ii,rules[ii]]

    x_train = x_train + stim_noise * np.random.randn(batch_size, n_steps, n_in)
    params['input_times'] = input_times
    params['output_times'] = output_times

    #plt.plot(range(len(x_train[0,:,0])), x_train[0,:,0])
    #plt.show()
    #plt.plot(range(len(y_train[0, :, 0])), y_train[0, :, 0])
    #plt.show()
    trial_info = dict()
    trial_info['rules'] = rules
    trial_info['dirs'] = dirs
    trial_info['cohs'] = cohs
    trial_info['stims'] = stims

    return x_train, y_train, mask, trial_info

def white_noise_test(sim,x_test):
    
    n_trials = x_test.shape[0]
    choice = np.zeros(n_trials)
    resp = np.zeros(n_trials)

    for ii in range(n_trials):
        o,s = sim.run_trial(x_test[ii,:,:],t_connectivity=False)
        resp[ii] = o[-1,0,:]
        choice[ii] = np.sign(resp[ii])
        
    mean_up = np.mean(x_test[choice==1,:,:],axis=0)
    mean_down = np.mean(x_test[choice==-1,:,:],axis=0)
    
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(mean_up)
    plt.title('Average Up')
    plt.subplot(1,2,2)
    plt.plot(mean_down)
    plt.title('Average Down')
    
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.bar([0,1],[np.mean(choice==1),np.mean(choice==-1)])
    plt.xticks([.35,1.45],['Up','Down'])
    plt.xlabel('Percent Up')
    plt.subplot(1,2,2)
    plt.hist(resp,20)
    plt.title('Response Histogram')
    plt.show()
    
    return mean_up,mean_down,choice,resp
        

def coherence_test(sim,cohs = [.2,.1,.05,.04,.02],n_hidden=50,sigma_in = 0):
    
    n_cohs = len(cohs)
    a = np.zeros([200,1])
    a[40:140] = 1
    o = np.zeros([200,n_cohs])
    s = np.zeros([200,n_hidden,n_cohs])
    ev = np.zeros([200,n_cohs])
    for ii,coh in enumerate(cohs): 
        inp = coh*a + sigma_in*np.random.randn(len(a),1)
        o_temp,s_temp = sim.run_trial(inp,t_connectivity=False)
        o[:,ii] = o_temp[:,0,:].flatten()
        s[:,:,ii] = s_temp[:,0,:]
        ev[:,ii] = np.cumsum(coh*a)

    
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(o)
    plt.title('output')
    
    plt.subplot(1,2,2)
    plt.plot(ev)
    plt.title('sum of evidence')
    
    plt.show()
    
    return o,s

        
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
    n_hidden  = 50
    n_out = 1
    #n_steps = 80 
    tau = 100.0 #As double
    dt = 20.0  #As double
    dale_ratio = None
    rec_noise = 0.0
    stim_noise = 0.1
    batch_size = 128
    #var_delay_length = 50
    cohs = [.01,.05,.1,.2,.4]
    rt_version = True
    
    #train params
    learning_rate = .0001
    training_iters = 300000
    display_step = 50
    
    weights_path = '../weights/rdm.npz'
    #weights_path = None
    
    params = set_params(n_in = n_in, n_out = n_out, n_steps = 200, coherences=cohs, 
                        stim_noise = stim_noise, rec_noise = rec_noise, L1_rec = 0, 
                        L2_firing_rate = 0, sample_size = 128, epochs = 100, N_rec = 50, 
                        dale_ratio=dale_ratio, tau=tau, dt = dt, task='n_back',rt_version=rt_version)
    
    generator = generate_train_trials(params)
    model = Model(params)
    sess = tf.Session()
    
    
    
    model.train(sess, generator, learning_rate = learning_rate, training_iters = training_iters,
                weights_path = weights_path,display_step=display_step)
    
    output_weights_path = weights_path
    
    'external weight intializer class'
    autapses = True
    w_initializer = WeightInitializer(params, output_weights_path[:-4] + '_init', autapses=autapses)
    input_weights_path = w_initializer.gen_weight_dict()
    params['load_weights_path'] = input_weights_path + '.npz'

    data = generator.next()
    
    
    #W = model.W_rec.eval(session=sess)
    #U = model.W_in.eval(session=sess)
    #Z = model.W_out.eval(session=sess)
    #brec = model.b_rec.eval(session=sess)
    #bout = model.b_out.eval(session=sess)
    
    #sim = Simulator(params, weights_path=weights_path)
    #output,states = sim.run_trial(data[0][0,:,:],t_connectivity=False)
    
    #x_test,y_test,mask = build_test_trials(params)
    #mup,mdown,choice,resp = white_noise_test(sim, x_test)
    #coh_out = coherence_test(sim, np.arange(-.2,.2,.01))

    #for i in range(5):
    #    trial = data[0][i,:,:]

    #    points = analysis.hahnloser_fixed_point(sim, trial)

    #    analysis.plot_states(states=states, I=points)

    
    sess.close()
