

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'viridis'

import numpy as np
import tensorflow as tf
from backend.networks import Model
#import backend.visualizations as V
from backend.simulation import Simulator


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

def relu(s):
    return np.maximum(s,0)

def principal_angle(A,B):
    ''' A = n x p
        B = n x q'''
    
    Qa, ra = np.linalg.qr(A)
    Qb, rb = np.linalg.qr(B)
    C = np.linalg.svd(Qa.T.dot(Qb))
    angles = np.arccos(C[1])
    
    return 180*angles/np.pi
    
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
    
def plot_single_trial(data,states,output):
    fig = plt.figure(figsize=(5,5))
    plt.subplot(3,1,1)
    plt.plot(output[:,0,:])
    plt.title('Out')
    plt.subplot(3,1,2)
    plt.plot(states[:,0,:])
    plt.title('State')
    plt.subplot(3,1,3)
    plt.plot(data[0][0,:,:])
    plt.title('Input')
    plt.tight_layout()
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
        
    plt.xlabel('Neuron')
    plt.title('Activity (fps) at end of Trial')
        
    return fig
    
def plot_outputs_by_input(s,data,weights,n=5):
    
    fig = plt.figure()
    colors = ['r','g','b','k','c']*10
    
    for ii in range(n): 
        out = np.maximum(s[-1,data[0][:,40,ii]>.2,:],0).dot(weights['W_out'].T) + weights['b_out']
        plt.plot(out.T,c=colors[np.mod(ii,5)],alpha=.4)
        
    response = np.argmax(relu(s[-1,:,:]).dot(weights['W_out'].T)+weights['b_out'],axis=1)

    inp = np.argmax(data[0][:,40,:],axis=1)
    accuracy = np.sum(inp==response)/float(len(inp))

    plt.xlabel('accuracy = ' + str(100*accuracy) + ' %')

    plt.title('Output as a function of Input')
    return fig

def pca_plot(n_in,s_long,s,inp,brec,n_reps=8):

    s_pca = demean(s_long[300,:,:])
    c_pca = np.cov(s_pca.T)
    evals,evecs = np.linalg.eig(c_pca)

    colors = ['r','g','b','k','c']*10
    fig = plt.figure()
    for ii in range(n_in):
        for jj in range(n_reps):
            plt.plot(s[:,inp==ii,:][:,jj,:].dot(evecs[:,0]),s[:,inp==ii,:][:,jj,:].dot(evecs[:,1]),c=colors[ii],alpha=.25)

    plt.plot(brec.dot(evecs[:,0]),brec.dot(evecs[:,1]),'kx')
    
    return fig

def plot_long_output_by_input(n_in,n_rec,s_long,weights):

    colors = ['r','g','b','k','c']*10
    fig = plt.figure(figsize=(8,1.5))
    for ii in range(n_in):
        plt.subplot(1,n_in,ii+1)
        response = relu(s_long[300,ii,:]).dot(weights['W_out'].T) + weights['b_out']
        max_real = np.max(np.linalg.eig(weights['W_rec']*(s_long[300,ii,:]>0)-np.eye(n_rec))[0].real)
        stable = max_real<0
        #print max_real
        if stable:
            plt.plot(response,c=colors[np.argmax(response)])
        else:
            plt.plot(response,'--',c=colors[np.argmax(response)])
          
        plt.tight_layout()
        
    return fig

def ablation_analysis(n_rec,n_in,weights,sim):

    abl_trial_steps = 1000
    W = weights['W_rec']
    t_cons = []

    colors = ['r','g','b','k','c']*10
    
    abl_in = np.zeros([abl_trial_steps,n_in*n_rec,n_in])
    for jj in range(n_rec):
        for ii in range(n_in):
            abl_in[10:80,ii+jj*n_in,ii] = 1
            mask = np.ones([n_rec,n_rec])
            mask[:,jj] = 0
            t_cons.append([mask])

    # plt.pcolormesh(abl_in[20,:,:])
    # plt.show()

    s_abl = np.zeros([abl_in.shape[0],abl_in.shape[1],W.shape[0]])
    for ii in range(n_in*n_rec):
        s_abl[:,ii,:] = sim.run_trial(abl_in[:,ii,:],t_connectivity=t_cons[ii]*abl_trial_steps)[1].reshape([abl_in.shape[0],W.shape[0]])

    count = 1
    outcome = np.zeros([n_rec,n_in])
    fig = plt.figure(figsize=(8,6))
    for ii in range(n_rec):
        for jj in range(n_in):
            plt.subplot(n_rec,n_in,count)
            response = relu(s_abl[300,count-1,:]).dot(weights['W_out'].T) + weights['b_out']
            stable = np.max(np.linalg.eig(W*(s_abl[300,ii,:]>0)-np.eye(n_rec))[0].real)<0
    #         in_part = np.sum(np.linalg.inv(np.eye(n_rec)-W*(s_abl[300,ii,:]>0)).dot(brec) != s_abl[300,ii,:]>0) == 0
            outcome[ii,jj] = np.argmax(response)
            if stable:
                plt.plot(response,c=colors[np.argmax(response)])
            else:
                plt.plot(response,'--',c=colors[np.argmax(response)])

            count += 1
    
    plt.tight_layout()
    
    return fig

def plot_structure_Wrec(W):
    N = W.shape[0]

    R = np.random.randn(N,N)/float(N)
    R = 1.1*R/np.max(np.abs(np.linalg.eig(R)[0]))
    
    #calculate the norm of trained rec matrix W and random gaussian matrix R
    normW = calc_norm(W)
    normR = calc_norm(R)
    min_norm = np.min([np.min(normW),np.min(normR)])
    max_norm = np.max([np.max(normW),np.max(normR)])
    xx_norm = np.linspace(min_norm,max_norm,50)
    histnormW, _ = np.histogram(normW,xx_norm)
    histnormR, _ = np.histogram(normR,xx_norm)
    
    #calculate hists for angles between columns
    
    angle_W = np.arccos(np.clip((W.T.dot(W))/np.outer(normW,normW),-1.,1.))
    angle_R = np.arccos(np.clip((R.T.dot(R))/np.outer(normR,normR),-1.,1.))
    min_val = np.min([np.min(angle_W),np.min(angle_R)])
    max_val = np.max([np.max(angle_W),np.max(angle_R)])
    xx = np.linspace(min_val,max_val,50)
    histW, bin_edgesW = np.histogram(angle_W[np.tril(np.ones_like(W),-1)>0],xx)
    histR, bin_edgesR = np.histogram(angle_R[np.tril(np.ones_like(R),-1)>0],xx)
    
    fig = plt.figure(figsize=(8,8))
    
    plt.subplot(3,2,1)
    plt.pcolormesh(W)
    plt.colorbar()
    plt.title('W')
    
    plt.subplot(3,2,2)
    plt.pcolormesh(R)
    plt.colorbar()
    plt.title('R')
    
    plt.subplot(3,2,3)
    plt.pcolormesh(angle_W)
    plt.colorbar()
    plt.title('$\measuredangle$ W')
    
    plt.subplot(3,2,4)
    plt.pcolormesh(angle_R)
    plt.colorbar()
    plt.title('$\measuredangle$ R')
    
    plt.subplot(3,2,5)
    plt.bar(xx[:-1],histW,width=bin_edgesW[1]-bin_edgesW[0])
    plt.bar(xx[:-1],-histR,width=bin_edgesR[1]-bin_edgesR[0],color='g')
    
    plt.legend(['W','Random'],fontsize=10,loc='lower left')
    plt.title('Hist of Angles')
    
    plt.subplot(3,2,6)
    plt.bar(xx_norm[:-1],histnormW,width=xx_norm[1]-xx_norm[0])
    plt.bar(xx_norm[:-1],-histnormR,width=xx_norm[1]-xx_norm[0],color='g')
    
    plt.legend(['W','Random'],fontsize=10,loc='lower left')
    plt.title('Hist of Norms')
    plt.tight_layout()
    
    return fig
    
def plot_dist_to_fp(s_long):
    fig = plt.figure()

    colors = ['r','g','b','k','c']*10

    part_dist = np.min(np.abs(s_long[300,:,:]),axis=1)
    plt.bar(np.arange(s_long.shape[1]),part_dist,color=colors[:s_long.shape[1]])
    plt.title('Distance to Nearest Partition')
    
    return fig
    
def plot_fp_partitions(s_long):
    
    fig = plt.figure(figsize=(8,2))
    plt.pcolormesh(s_long[300,:,:]>0,cmap='gray')
    plt.colorbar()
    
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

    params['sample_size'] = 2000
    generator = generate_train_trials(params)
    weights = np.load(weights_path)
    
    W = weights['W_rec']
    Win = weights['W_in']
    Wout = weights['W_out']
    brec = weights['b_rec'] 
    
    #Generate Input Data
    data = generator.next()
    #Find Input/Target One-Hot
    inp = np.argmax(data[0][:,40,:],axis=1)
    
    sim = Simulator(params, weights_path=weights_path)
    output,states = sim.run_trial(data[0][0,:,:],t_connectivity=False)
    
    n_in = n_out = data[0].shape[2]
    n_rec = W.shape[0]
    
    #generate trials
    s = np.zeros([data[0].shape[1],data[0].shape[0],W.shape[0]])
    for ii in range(data[0].shape[0]):
        s[:,ii,:] = sim.run_trial(data[0][ii,:,:],t_connectivity=False)[1].reshape([data[0].shape[1],W.shape[0]])
    
    #generate long duration trials
    long_in = np.zeros([10000,n_in,n_in])
    for ii in range(n_in):
        long_in[10:80,ii,ii] = 1

    s_long = np.zeros([long_in.shape[0],long_in.shape[1],W.shape[0]])
    for ii in range(n_in):
        s_long[:,ii,:] = sim.run_trial(long_in[:,ii,:],t_connectivity=False)[1].reshape([long_in.shape[0],W.shape[0]])
    
    #Figure 0 (Plot Params)
    fig0 = plot_params(original_params)
    pp.savefig(fig0)

    #Figure 1 (Single Trial (Input Output State))
    fig1 = plot_single_trial(data,states,output)
    pp.savefig(fig1)    
    
    #Figure 2 (plot fixed points - activity at end of trial)
    fig2 = plot_fps_vs_activity(s,W,brec)
    pp.savefig(fig2)
    
    #Figure 3 (Plot output activity)
    try:
        fig3 = plot_outputs_by_input(s,data,weights,n=Win.shape[1])
        pp.savefig(fig3)
    except Exception:
        pass
    
    #Figure 4 (Plot 2D PCA projection)
    fig4 = pca_plot(n_in,s_long,s,inp,brec)
    pp.savefig(fig4)
    
    #Figure5 (Plot Long Output)
    fig5 = plot_long_output_by_input(n_in,n_rec,s_long,weights)
    pp.savefig(fig5)
    
    #Figure6 (Plot ablation analysis)
    fig6 = ablation_analysis(n_rec,n_in,weights,sim)
    pp.savefig(fig6)
    
    #Figure7 (Plot W Structure)
    fig7 = plot_structure_Wrec(W)
    pp.savefig(fig7)
    
    #Figure8 (Bar Plot of distance to nearest partition)
    fig8 = plot_dist_to_fp(s_long)
    pp.savefig(fig8)
    
    fig9 = plot_fp_partitions(s_long)
    pp.savefig(fig9)
    
    
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
                    sample_size = 128, epochs = 100, N_rec = n_rec, dale_ratio=dale_ratio, tau=tau, dt = dt, task='n_fixed')
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
