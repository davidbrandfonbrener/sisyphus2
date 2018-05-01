

import numpy as np
import tensorflow as tf
from backend.networks import Model
from backend import analysis
#import backend.visualizations as V
from backend.simulation import Simulator
import matplotlib.pyplot as plt


# Builds a dictionary of parameters that specifies the information
# about an instance of this specific task
def set_params(n_in = 1, n_out = 1, n_steps = 200, coherences=[.5], stim_noise = 0, rec_noise = 0, L1_rec = 0, L2_firing_rate = 0,
                    sample_size = 128, epochs = 100, N_rec = 50, dale_ratio=0.8, tau=100.0, dt = 10.0, biases = True,
               task='n_back', rt_version=False):
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
    stim_noise = params['stim_noise']
    batch_size = int(params['sample_size'])
    rt_version = params['rt_version']
    coherences = params['coherences']

    input_times = np.zeros([batch_size, n_in], dtype=np.int)
    output_times = np.zeros([batch_size, n_out], dtype=np.int)

    x_train = np.zeros([batch_size,n_steps,n_in])
    y_train = np.zeros([batch_size,n_steps,n_out])
    mask = np.ones((batch_size, n_steps, n_in))
    
    stim_time = range(40,140)
    if rt_version:
        out_time = range(50,200)
    else:
        out_time = range(160,200)

    dirs = np.random.choice([-1,1],replace=True,size=(batch_size))
    cohs = np.random.choice(coherences,replace=True,size=(batch_size))
    stims = dirs*cohs;
    
    for ii in range(batch_size):
        x_train[ii,stim_time,0] = stims[ii]
        y_train[ii,out_time,0] = dirs[ii]

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
    stim_noise = params['stim_noise']
    rt_version = params['rt_version']
    batch_size = 1000
    coherences = [0.]

    x_test = np.zeros([batch_size,n_steps,n_in])
    y_test = np.zeros([batch_size,n_steps,n_out])
    mask = np.ones((batch_size, n_steps, n_in))
    
    stim_time = range(40,140)
    if rt_version:
        out_time = range(50,200)
    else:
        out_time = range(160,200)

    dirs = np.random.choice([-1,1],replace=True,size=(batch_size))
    cohs = np.random.choice(coherences,replace=True,size=(batch_size))
    stims = dirs*cohs;
    
    for ii in range(batch_size):
        x_test[ii,stim_time,0] = stims[ii]
        y_test[ii,out_time,0] = dirs[ii]

    x_test = x_test + stim_noise * np.random.randn(batch_size, n_steps, n_in)
    
    return x_test, y_test, mask

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

    data = generator.next()
    
    
    #W = model.W_rec.eval(session=sess)
    #U = model.W_in.eval(session=sess)
    #Z = model.W_out.eval(session=sess)
    #brec = model.b_rec.eval(session=sess)
    #bout = model.b_out.eval(session=sess)
    
    sim = Simulator(params, weights_path=weights_path)
    output,states = sim.run_trial(data[0][0,:,:],t_connectivity=False)
    
    x_test,y_test,mask = build_test_trials(params)
    mup,mdown,choice,resp = white_noise_test(sim, x_test)
    coh_out = coherence_test(sim, np.arange(-.2,.2,.01))

    for i in range(5):
        trial = data[0][i,:,:]

        points = analysis.hahnloser_fixed_point(sim, trial)

        analysis.plot_states(states=states, I=points)

    
    sess.close()
