
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from backend.networks import Model
import backend.visualizations as V
from backend.simulation_tools import Simulator
from scipy.spatial.distance import pdist, squareform



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
#         fr = ((fr-np.mean(fr,axis=0))/np.std(fr,axis=0))
        fr = ((fr-np.min(fr,axis=0))/(1e-10+np.max(fr,axis=0)-np.min(fr,axis=0)))
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

def principal_angle(A,B):
    ''' A = n x p
        B = n x q'''
    
    Qa, ra = np.linalg.qr(A)
    Qb, rb = np.linalg.qr(B)
    C = np.linalg.svd(Qa.T.dot(Qb))
    angles = np.arccos(C[1])
    
    return 180*angles/np.pi

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


def plot_input_output_angles(Win,W,Wout,brec):

    fig = plt.figure(figsize=(10,4))
    plt.subplot(2,3,1)
    plt.pcolormesh(gen_angle(Win,Win))
    plt.colorbar()
    plt.title('angle $W_{in}$ by $W_{in}$')
    plt.subplot(2,3,2)
    plt.pcolormesh(gen_angle(Wout.T,Wout.T))
    plt.colorbar()
    plt.title('angle $W_{out}$ by $W_{out}$')
    plt.subplot(2,3,3)
    plt.pcolormesh(gen_angle(Win,Wout.T))
    plt.colorbar()
    plt.title('angle $W_{in}$ by $W_{out}$')
    plt.subplot(2,3,4)
    plt.bar(range(3),calc_norm(Win))
    plt.xticks([])
    plt.xlim(-.2,3)
    plt.xlabel('Win')
    plt.ylabel('Norm')
    plt.subplot(2,3,5)
    plt.bar(range(2),calc_norm(Wout.T))
    plt.xticks([])
    plt.xlim(-.2,2)
    plt.xlabel('Wout')
    plt.ylabel('Norm')

    plt.tight_layout()
    
    return fig
    
def plot_biclustered_weights(W):
    import sklearn.cluster
    model = sklearn.cluster.bicluster.SpectralBiclustering(n_clusters=4, method='log', random_state=0)
    model.fit(W)
    
    fit_W = W[np.argsort(model.row_labels_)]
    fit_W = fit_W[:, np.argsort(model.column_labels_)]
                  
    fig = plt.figure(figsize=(6,4))
    plt.pcolormesh(fit_W,cmap='viridis')
    plt.colorbar()
    plt.axes().set_aspect('equal')
    plt.title('Biclustering of Wrec')
    
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
    
def plot_principal_angles(W,s,data):
    masks = s[:,0,:].T>0
    max_ev = np.zeros(data[0].shape[1])
    
    pos = []
    neg = []
    leading = []
    for ii in range(data[0].shape[1]):
        evals,evecs = np.linalg.eig(W*masks[:,ii]-np.eye(100))
        max_ev[ii] = np.max(evals.real)
        pos.append(evecs[:,evals>0])
        neg.append(evecs[:,evals<0])
        leading.append(evecs[:,np.argsort(np.abs(evals.real))[:10]]) #.reshape([100,2]))
        
    
    xx = np.arange(0,data[0].shape[1],1)
    pa = np.zeros([len(xx),len(xx)])
    
    basis = leading
    
    for ii,pre in enumerate(xx):
        for jj,post in enumerate(xx):
            if basis[pre].shape[1]*basis[post].shape[1]>0:
                pas = principal_angle(basis[pre],basis[post])
                pa[ii,jj] = np.nanmean(pas)
            else:
                pa[ii,jj] = 0.
    
    fig = plt.figure()        
    plt.pcolormesh(pa,vmin=0,vmax=90)
    plt.colorbar()
    plt.ylim([0,pa.shape[0]])
    plt.xlim([0,pa.shape[1]])
    
    plt.axes().set_aspect('equal')
    plt.title('Principal Angle Analysis')
    plt.xlabel('Time')
    plt.ylabel('Time')
    
    return fig
    
def plot_long_term_state(sim):
    d = .01*np.random.randn(2000,3)
    d[50:60,0] = 1.
    o_in0,s_in0 = sim.run_trial(d,t_connectivity=False)
    
    
    d[50:60,1] = 1.
    o_in1,s_in1 = sim.run_trial(d,t_connectivity=False)
    
    d = .01*np.random.randn(2000,3)
    d[50:60,0] = 1.
    d[150:160,2] = 1.
    o_go0,s_go0 = sim.run_trial(d,t_connectivity=False)
    
    
    d[50:60,1] = 1.
    d[150:160,2] = 1.
    o_go1,s_go1 = sim.run_trial(d,t_connectivity=False)
    
    fig = plt.figure(figsize=(8,6))
    
    plt.subplot(4,2,1)
    plt.plot(s_in1[:500,0,:]);
    plt.title('Long Input 1')
    plt.subplot(4,2,3)
    plt.plot(s_in0[:500,0,:]);
    plt.title('Long Input 2')
    plt.subplot(4,2,5)
    plt.plot(s_in0[:500,0,:] - s_in1[:500,0,:]);
    plt.title('Difference')
    plt.subplot(4,2,7)
    plt.plot(o_in1[:500,0,:]);
    plt.plot(o_in0[:500,0,:]);
    plt.title('Output')
    
    
    plt.subplot(4,2,2)
    plt.plot(s_go0[:500,0,:]);
    plt.title('Long Input 1 + Go Cue')
    plt.subplot(4,2,4)
    plt.plot(s_go1[:500,0,:]);
    plt.title('Long Input 2 + Go Cue')
    plt.subplot(4,2,6)
    plt.plot(s_go0[:500,0,:] - s_go1[:500,0,:]);
    plt.title('Difference')
    plt.subplot(4,2,8)
    plt.plot(o_go0[:500,0,:]);
    plt.plot(o_go1[:500,0,:]);
    plt.title('Output')
    
    plt.tight_layout()
    
    return fig
    
def plot_state_pcs(sim):
    d = .01*np.random.randn(2000,3)
    d[50:60,0] = 1.
    o_in0,s_in0 = sim.run_trial(d,t_connectivity=False)
    
    
    d[50:60,1] = 1.
    o_in1,s_in1 = sim.run_trial(d,t_connectivity=False)
    
    d = .01*np.random.randn(2000,3)
    d[50:60,0] = 1.
    d[150:160,2] = 1.
    o_go0,s_go0 = sim.run_trial(d,t_connectivity=False)
    
    
    d[50:60,1] = 1.
    d[150:160,2] = 1.
    o_go1,s_go1 = sim.run_trial(d,t_connectivity=False)
    
    s_pca = np.concatenate((s_go0[:500,0,:],s_go1[:500,0,:]),axis=0)
    s_pca = demean(s_pca)
    c_pca = np.cov(s_pca.T)
    evals,evecs = np.linalg.eig(c_pca)
    
    fig = plt.figure()
    plt.plot(s_go0[:,0,:].dot(evecs[:,0:1]),s_go0[:,0,:].dot(evecs[:,1:2]),'g',alpha=.5)
    plt.plot(s_go1[:,0,:].dot(evecs[:,0:1]),s_go1[:,0,:].dot(evecs[:,1:2]),'b',alpha=.5)
    plt.plot(s_in0[:,0,:].dot(evecs[:,0:1]),s_in0[:,0,:].dot(evecs[:,1:2]),'c',alpha=.5)
    plt.plot(s_in1[:,0,:].dot(evecs[:,0:1]),s_in1[:,0,:].dot(evecs[:,1:2]),'r',alpha=.5)
    
    plt.plot(s_go1[:,0,:].dot(evecs[:,0:1])[0],s_go1[:,0,:].dot(evecs[:,1:2])[0],'kx',markersize=10)
    
    plt.plot(s_go0[:,0,:].dot(evecs[:,0:1])[49],s_go0[:,0,:].dot(evecs[:,1:2])[49],'og',markersize=5)
    plt.plot(s_go0[:,0,:].dot(evecs[:,0:1])[149],s_go0[:,0,:].dot(evecs[:,1:2])[149],'og',markersize=5)
    
    plt.plot(s_go1[:,0,:].dot(evecs[:,0:1])[49],s_go1[:,0,:].dot(evecs[:,1:2])[49],'xb',markersize=8)
    plt.plot(s_go1[:,0,:].dot(evecs[:,0:1])[149],s_go1[:,0,:].dot(evecs[:,1:2])[149],'xb',markersize=8)
    
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    
    plt.legend(['in_go_0','in_go_1','in_0','in_1'],loc='lower left',fontsize=8)
    
    return fig
    
def plot_hamming_dist(s,W,brec):
    masks = s[:,0,:].T>0
    x_hat = np.zeros(masks.shape)

    for ii in range(masks.shape[1]):
        Weff = W*masks[:,ii]
        x_hat[:,ii] = np.linalg.inv(np.eye(100)-Weff).dot(brec)
    
    fig = plt.figure()
    plt.pcolormesh(squareform(pdist(np.sign(x_hat[:,:]).T,metric='hamming'))) #,vmax=.3)
    plt.colorbar()
    plt.ylim([0,x_hat.shape[1]])
    plt.xlim([0,x_hat.shape[1]])
    
    plt.axes().set_aspect('equal')
    plt.title('Hamming Distance Between Putative FPs')
    plt.ylabel('Time')
    plt.xlabel('Time')
    
    return fig
    
def plot_long_delayed_go_cue(sim):
    fig = plt.figure(figsize=(5,8))
    
    d = .01*np.random.randn(2000,3)
    d[50:60,0] = 1.
    # d[100:110,1] = 2.
    # d[200:210,0] = 2.
    # d[300:310,1] = 2.
    d[800:810,2] = 1.
    d[1300:1310,2] = 1.
    d[1800:1810,2] = 1.
    
    o_long,s_long = sim.run_trial(d,t_connectivity=False)
    
    plt.subplot(4,1,1)
    plt.plot(d)
    plt.title('Input')
    
    plt.subplot(4,1,2)
    plt.plot(o_long[:,0,:]);
    plt.title('Output')
    
    d = .01*np.random.randn(2000,3)
    d[50:60,1] = 1.
    # d[100:110,1] = 2.
    # d[200:210,0] = 2.
    # d[300:310,1] = 2.
    d[800:810,2] = 1.
    d[1300:1310,2] = 1.
    d[1800:1810,2] = 1.
    
    o_long,s_long = sim.run_trial(d,t_connectivity=False)
    
    plt.subplot(4,1,3)
    plt.plot(d)
    plt.title('Input')
    
    plt.subplot(4,1,4)
    plt.plot(o_long[:,0,:]);
    plt.title('Output')
    
    plt.tight_layout()
    
    return fig
    
def plot_biases(weights):
    brec = weights['b_rec']
    bout = weights['b_out']
    win = weights['W_in']
    wout = weights['W_out']
    wrec = weights['W_rec']

    fig = plt.figure(figsize=(10,6))
    plt.subplot(2,3,1)
    plt.hist(brec)
    plt.title('brec')
    plt.tick_params(labelsize=8)
    
    plt.subplot(2,3,2)
    plt.bar([0,1],bout)
    plt.title('bout')
    plt.tick_params(labelsize=8)
    
    plt.subplot(2,3,4)
    plt.plot(win[:,0],win[:,1],'.')
    plt.xlabel('W_in 0')
    plt.ylabel('W_in 1')
    plt.tick_params(labelsize=8)
    
    plt.subplot(2,3,5)
    plt.plot(wout[0,:],wout[1,:],'.')
    plt.xlabel('W_out 0')
    plt.ylabel('W_out 1')
    plt.tick_params(labelsize=8)
    
    plt.subplot(2,3,3)
    plt.hist(np.diag(wrec))
    plt.title('diag(W_rec)')
    plt.tick_params(labelsize=8)
    
    plt.subplot(2,3,6)
    plt.plot(brec,np.diag(wrec),'.')
    plt.xlabel('b_rec')
    plt.ylabel('diag(W_rec)')
    plt.tick_params(labelsize=8)
    
    plt.tight_layout()
    
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
    
    s = np.zeros([data[0].shape[1],data[0].shape[0],100])
    for ii in range(data[0].shape[0]):
        s[:,ii,:] = sim.run_trial(data[0][ii,:,:],t_connectivity=False)[1].reshape([data[0].shape[1],100])
    
    #Figure 0 (Plot Params)
    fig0 = plot_params(original_params)
    pp.savefig(fig0)    
    
    #Figure 1 (Single Trial (Input Output State))
    fig1 = plot_single_trial(data,states,output)
    pp.savefig(fig1)
    
    #Figure 2 (Plot structural measures of W against random matrix R)
    fig2 = plot_structure_Wrec(W)
    pp.savefig(fig2)
    
    #Figure 3 (Stupid Figure where activity is sorted by time of max firing rate)
    fig3 = plt.figure(figsize=(7,3))
    plot_by_max(s[:,0,:])
    plt.xlabel('Time')
    plt.ylabel('Neuron')
    pp.savefig(fig3)
    
    #Figure 4 (Principal Angle Analysis)
    fig4 = plot_principal_angles(W,s,data)
    pp.savefig(fig4)

    #Figure 5 Plot long term state activity for in, and in+go conditions
    fig5 = plot_long_term_state(sim)
    pp.savefig(fig5)
    
    #Figure 6 Plot PC projection
    fig6 = plot_state_pcs(sim)
    pp.savefig(fig6)
    
    #Figure 7 Hamming Distance btw Putative Fixed Points
    fig7 = plot_hamming_dist(s,W,brec)
    pp.savefig(fig7)
    
    #Figure 8 Plot long delayed go cue
    fig8 = plot_long_delayed_go_cue(sim)
    pp.savefig(fig8)
    
    #Figure 9 Plot angles between input mapping and output mapping
    fig9 = plot_input_output_angles(Win,W,Wout,brec)    
    pp.savefig(fig9)
    
    #Figure 10 Plot biclustered recurrent weights
    fig10 = plot_biclustered_weights(W)
    pp.savefig(fig10)
    
    #Figure 11 Plot biases
    fig11 = plot_biases(weights)
    pp.savefig(fig11)
    
    
    pp.close()
            
if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', help="task name", type=str)
    parser.add_argument('fig_directory',help="where to save figures")
    parser.add_argument('weights_path',help="where to save weights")
    parser.add_argument('-m','--mem_gap', help="supply memory gap length", type=int,default=50)
    parser.add_argument('-v','--var_delay', help="supply variable memory gap delay", type=int,default=0)
    parser.add_argument('-r','--rec_noise', help ="recurrent noise", type=float,default=0.0)
    parser.add_argument('-t','--training_iters', help="training iterations", type=int,default=300000)
    parser.add_argument('-ts','--task',help="task type",default='memory_saccade')
    args = parser.parse_args()
    
    #run params
    run_name = args.run_name
    fig_directory = args.fig_directory
    
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
    weights_path = args.weights_path
    
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
    
    analysis_and_write(params,weights_path,fig_directory,run_name)

#    data = generator.next()
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
#    s = np.zeros([states.shape[0],batch_size,n_hidden])
#    for ii in range(batch_size): 
#        s[:,ii,:] = sim.run_trial(data[0][ii,:,:],t_connectivity=False)[1].reshape([states.shape[0],n_hidden])
    
    sess.close()



