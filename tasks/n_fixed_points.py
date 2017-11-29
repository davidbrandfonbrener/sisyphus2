

import numpy as np
import tensorflow as tf
from backend.networks import Model
#import backend.visualizations as V
from backend.simulation_tools import Simulator
import matplotlib.pyplot as plt


# Builds a dictionary of parameters that specifies the information
# about an instance of this specific task
def set_params(n_in = 5, n_out = 5, n_fixed_points = 5, n_steps = 200, stim_noise = 0, rec_noise = 0, L1_rec = 0, L2_firing_rate = 0,
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
    #params['N_fixed_points']   = n_fixed_points
    params['biases']           = biases

    return params

# This generates the training data for our network
# It will be a set of input_times and output_times for when we expect input
# and when the corresponding output is expected
def build_train_trials(params):
    n_in = params['N_in']
    n_out = n_in
    n_steps = params['N_steps']
    #input_wait = params['input_wait']
    #mem_gap = params['mem_gap']
    #stim_dur = params['stim_dur']
    #out_dur = params['out_dur']
    #var_delay_length = params['var_delay_length']
    n_fixed_points = n_in
    stim_noise = params['stim_noise']
    batch_size = int(params['sample_size'])
    #task = params['task']
    
    fixed_pts = np.random.randint(low=0,high=n_fixed_points,size=batch_size)

    input_times = np.zeros([batch_size, n_in], dtype=np.int)
    output_times = np.zeros([batch_size, n_out], dtype=np.int)

    x_train = np.zeros([batch_size,n_steps,n_in])
    y_train = np.zeros([batch_size,n_steps,n_out])
    mask = np.ones((batch_size, n_steps, n_in))
    
    stim_time = range(10,80)
    out_time = range(60,n_steps)
    for ii in range(batch_size):
        x_train[ii,stim_time,fixed_pts[ii]] = 1.
        y_train[ii,out_time,fixed_pts[ii]] = 1.

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
        
def calc_norm(A):
    return np.sqrt(np.sum(A**2,axis=0))
    
def demean(s):
    return s-np.mean(s,axis=0)
    
def gen_angle(W,U):
    normW = calc_norm(W)
    normU = calc_norm(U)
    return np.arccos(np.clip((W.T.dot(U))/np.outer(normW,normU),-1.,1.))
    
def plot_params(params):
    params['input_times'] = []
    params['output_times'] = []
    ordered_keys = sorted(params)
    fig = plt.figure(figsize=(8,11),frameon=False); 
    for ii in range(len(params)): 
        item = ordered_keys[ii] + ': ' + str(params[ordered_keys[ii]])
        plt.text(.1,.9-.9/len(params)*ii,item)
    ax = plt.gca()
    ax.axis('off')
        
    return fig
    
        
def plot_fps_vs_activity(s,W,brec):
    
    fig = plt.figure(figsize=(4,8))
    
    for ii in range(5):
        plt.subplot(5,1,ii+1)
        Weff = W*(s[-1,ii,:]>0)
        fp = np.linalg.inv(np.eye(s.shape[2])-Weff).dot(brec)
        max_real = np.max(np.linalg.eig(Weff-np.eye(s.shape[2]))[0].real)
        plt.plot(s[60:,ii,:].T,c='c',alpha=.05)
        if max_real<0:
            plt.plot(fp,'k--')
        else:
            plt.plot(fp,'r--')
        plt.axhline(0,c='k')
        
    return fig
    
def plot_outputs_by_input(s,data,Z,n=5):
    
    fig = plt.figure()
    colors = ['r','g','b','k','c']
    
    for ii in range(n): 
        out = np.maximum(s[-1,data[0][:,40,ii]>.2,:],0).dot(Z.T).T
        plt.plot(out,c=colors[np.mod(ii,5)],alpha=.4)

    return fig
    
def analysis_and_write(params,weights_path,fig_directory,run_name,no_rec_noise=True):
    
    from matplotlib.backends.backend_pdf import PdfPages
    import os
    import copy
    
    original_params = copy.deepcopy(params)
    
    if no_rec_noise:
        params['rec_noise'] = 0.0
    
    try:
        os.stat(fig_directory)
    except:
        os.mkdir(fig_directory)
        
    pp = PdfPages(fig_directory + '/' + run_name + '.pdf')

    generator = generate_train_trials(params)
    weights = np.load(weights_path)
    
    W = weights['W_rec']
    Win = weights['W_in']
    Wout = weights['W_out']
    brec = weights['b_rec'] 
    
    data = generator.next()
    sim = Simulator(params, weights_path=weights_path)
    output,states = sim.run_trial(data[0][0,:,:],t_connectivity=False)
    
    s = np.zeros([data[0].shape[1],data[0].shape[0],W.shape[0]])
    for ii in range(data[0].shape[0]):
        s[:,ii,:] = sim.run_trial(data[0][ii,:,:],t_connectivity=False)[1].reshape([data[0].shape[1],W.shape[0]])
    
    #Figure 0 (Plot Params)
    fig0 = plot_params(original_params)
    pp.savefig(fig0)    
    
    #Figure 1 (Single Trial (Input Output State))
    fig1 = plot_fps_vs_activity(s,W,brec)
    pp.savefig(fig1)
    
    #Figure 2 (Plot structural measures of W against random matrix R)
    fig2 = plot_outputs_by_input(s,data,Wout,n=Win.shape[1])
    pp.savefig(fig2)
    
    
    pp.close()
        
if __name__ == "__main__":
    
    import time
    
    start_time = time.time()
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', help="task name", type=str)
    parser.add_argument('fig_directory',help="where to save figures")
    parser.add_argument('weights_path',help="where to save weights")
    parser.add_argument('-fp', '--n_fps', help="number of fixed points", type=int,default=5)
    parser.add_argument('-nr','--n_rec', help="number of hidden units", type=int,default=10)
    parser.add_argument('-i','--initialization', help ="initialization of Wrec", type=str,default='gauss')
    parser.add_argument('-r','--rec_noise', help ="recurrent noise", type=float,default=0.01)
    parser.add_argument('-t','--training_iters', help="training iterations", type=int,default=300000)
    parser.add_argument('-ts','--task',help="task type",default='fixed_point')
    args = parser.parse_args()
    
    #run params
    run_name = args.run_name
    fig_directory = args.fig_directory
    
    n_in = n_out = args.n_fps
    n_rec = args.n_rec
    
    #model params
    #n_in = n_out = 5 #number of fixed points
    #n_rec = 10 
    #n_steps = 80 
    tau = 100.0 #As double
    dt = 20.0  #As double
    dale_ratio = 0
    rec_noise = args.rec_noise
    stim_noise = 0.1
    batch_size = 128 #256
    #var_delay_length = 50
    
    n_back = 0
    
    #train params
    learning_rate = .0001 
    training_iters = args.training_iters
    display_step = 200
    
    #weights_path = '../weights/n_fps6by8_1.npz'
    save_weights_path = args.weights_path
    
    params = set_params(n_in = n_in, n_out = n_out, n_steps = 300, stim_noise = stim_noise, rec_noise = rec_noise, L1_rec = 0, L2_firing_rate = 0,
                    sample_size = 128, epochs = 100, N_rec = n_rec, dale_ratio=dale_ratio, tau=tau, dt = dt, task='n_back')
    generator = generate_train_trials(params)
    #model = Model(n_in, n_hidden, n_out, n_steps, tau, dt, dale_ratio, rec_noise, batch_size)
    model = Model(params)
    sess = tf.Session()
    
    
    model.train(sess, generator, learning_rate = learning_rate, training_iters = training_iters, save_weights_path = save_weights_path)
    #print('second training')
    #model.train(sess, generator, learning_rate = learning_rate, training_iters = training_iters, weights_path = weights_path, initialize_variables=False)

    analysis_and_write(params,save_weights_path,fig_directory,run_name)
    
#    data = generator.next()
#    inp = np.argmax(data[0][:,40,:],axis=1)
#    #output,states = model.test(sess, input, weights_path = weights_path)
#    
#    
#    W = model.W_rec.eval(session=sess)
#    U = model.W_in.eval(session=sess)
#    Z = model.W_out.eval(session=sess)
#    brec = model.b_rec.eval(session=sess)
#    bout = model.b_out.eval(session=sess)
#    
#    sim = Simulator(params, weights_path=weights_path)
#    output,states = sim.run_trial(data[0][0,:,:],t_connectivity=False)
#    
#    s = np.zeros([data[0].shape[1],data[0].shape[0],n_rec])
#    for ii in range(data[0].shape[0]):
#        s[:,ii,:] = sim.run_trial(data[0][ii,:,:],t_connectivity=False)[1].reshape([data[0].shape[1],n_rec])
      
    dur = time.time()-start_time
    print('runtime: '+ str(int(dur/60)) + ' min, ' + str(int(np.mod(dur,60))) + ' sec')
    
    sess.close()
