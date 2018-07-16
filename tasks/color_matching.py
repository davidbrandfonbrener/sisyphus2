

import numpy as np
import tensorflow as tf
from backend.networks import Model
from backend.simulation_tools import Simulator
import matplotlib.pyplot as plt


# Builds a dictionary of parameters that specifies the information
# about an instance of this specific task
def set_params(n_in = 2, n_out = 2, n_steps = 240, coherences=[.5], stim_noise = 0, rec_noise = 0, L1_rec = 0, L2_firing_rate = 0, reward_version = False,
                    sample_size = 128, epochs = 100, N_rec = 50, dale_ratio=0.8, tau=100.0, dt = 10.0, biases = False, task='n_back', rt_version=False):
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
    params['reward_version']   = reward_version

    return params

# This generates the training data for our network
# It will be a set of input_times and output_times for when we expect input
# and when the corresponding output is expected
def build_train_trials(params):
    n_in = params['N_in']
    n_out = params['N_out']
    n_steps = params['N_steps']
    stim_noise = params['stim_noise']
    batch_size = int(params['N_batch'])
    coherences = params['coherences']

    input_times = np.zeros([batch_size, n_in], dtype=np.int)
    output_times = np.zeros([batch_size, n_out], dtype=np.int)

    x_train = np.zeros([batch_size,n_steps,n_in])
    y_train = .1*np.ones([batch_size,n_steps,n_out])
    mask = np.ones((batch_size, n_steps, n_out))
    
    stim_times = [range(50,200), range(90,200), range(130,200)]
    out_times = [range(50,200), range(90,200), range(130,200)]

    timing = np.random.randint(low=0,high=3,size=(batch_size))
    dirs = np.random.choice([-1,1],replace=True,size=(batch_size))
    cohs = np.random.choice(coherences,replace=True,size=(batch_size))
    stims = dirs*cohs;
    
    for ii in range(batch_size):
        x_train[ii,stim_times[timing[ii]],0] = stims[ii]
        if dirs[ii]>0:
            y_train[ii,out_times[timing[ii]],0] = 1. #dirs[ii]
            if params['reward_version']:
                mask[ii,:,:]*=2.
        else:
            y_train[ii,out_times[timing[ii]],1] = 1. 
        mask[ii,out_times[timing[ii]][-1]:,:] = 0

    x_train = x_train + stim_noise * np.random.randn(batch_size, n_steps, n_in)
    params['input_times'] = input_times
    params['output_times'] = output_times
    return x_train, y_train, mask
    

def generate_train_trials(params):
    while 1 > 0:
        yield build_train_trials(params)
        
        
def build_noise_trials(params):
    
    n_in = params['N_in']
    n_out = params['N_out']
    n_steps = params['N_steps']
    stim_noise = params['stim_noise']
    batch_size = 1000
    coherences = [0.]

    x_test = np.zeros([batch_size,n_steps,n_in])
    y_test = np.zeros([batch_size,n_steps,n_out])
    mask = np.ones((batch_size, n_steps, n_in))
    
    stim_times = [range(50,170), range(50,170), range(50,170)]
    out_times = [range(50,120), range(90,160), range(130,200)]

    timing = np.random.randint(low=0,high=3,size=(batch_size))
    dirs = np.random.choice([-1,1],replace=True,size=(batch_size))
    cohs = np.random.choice(coherences,replace=True,size=(batch_size))
    stims = dirs*cohs;
    
    for ii in range(batch_size):
        x_test[ii,stim_times[timing[ii]],0] = stims[ii]
        y_test[ii,out_times[timing[ii]],0] = dirs[ii]
        mask[ii,out_times[timing[ii]][-1]:,0] = 0
        if dirs[ii] == 1:
            mask *= 2


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
    
def state_tensor_decomposition(states,rank=3):
    import tensorly as tly
    from tensorly.decomposition import parafac
    
    s_tens = tly.tensor(states)
    f = parafac(s_tens,rank=rank)
    factors = []
    
    for ii in range(len(f)):
        factors.append(tly.to_numpy(f[ii]))
    
    return factors

def get_rts(out,thresh=.4):

    n_trials = out.shape[1]
    n_steps = out.shape[0]

    rts = -np.ones(n_trials).astype(int)  
    choice = np.zeros(n_trials)
    for ii in range(n_trials): 
        cross = np.where(np.abs(out[:,ii])>thresh)
        if len(cross[0]) != 0:
            rts[ii] = np.min(cross)
            choice[ii] = np.sign(out[rts[ii],ii])
            
    return rts,choice
    
def plot_by_max(state,norm=True,thresh=.001):
    fr = np.maximum(state,thresh)
    if norm:
        #fr = ((fr-np.mean(fr,axis=0))/np.std(fr,axis=0))
        fr = fr/np.max(fr,axis=0)
    idx = np.argsort(np.argmax(fr,axis=0))
    plt.pcolormesh(fr[:,idx].T)
    plt.colorbar()
    plt.xlim([0,np.shape(fr)[0]])
    
def principal_angle(A,B):
    ''' A = n x p
        B = n x q'''
    
    Qa, ra = np.linalg.qr(A)
    Qb, rb = np.linalg.qr(B)
    C = np.linalg.svd(Qa.T.dot(Qb))
    angles = np.arccos(C[1])
    
    return angles

        
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
    dt = 10.0  #As double
    dale_ratio = 0
    rec_noise = 0.0
    stim_noise = 0.4
    batch_size = 256
    #var_delay_length = 50
    cohs = [.05,.08,.1,.2,.3,.4,.8,1.]
    
    #train params
    learning_rate = .0001
    training_iters = 200000
    display_step = 50
    
    weights_path = '../weights/color_matching3.npz'
    #weights_path = None
    
    params = set_params(n_in = n_in, n_out = n_out, n_steps = 240, coherences=cohs, 
                        stim_noise = stim_noise, rec_noise = rec_noise, L1_rec = 0, 
                        L2_firing_rate = 0, sample_size = 128, epochs = 100, N_rec = 50, 
                        dale_ratio=dale_ratio, tau=tau, dt = dt, task='color_matching')
    
    generator = generate_train_trials(params)
    #model = Model(n_in, n_hidden, n_out, n_steps, tau, dt, dale_ratio, rec_noise, batch_size)
    model = Model(params)
    sess = tf.Session()
    
    
    
    model.train(sess, generator, learning_rate = learning_rate, training_iters = training_iters, 
                weights_path = weights_path,display_step=display_step)

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
    
    n_noise = 1000
    s_noise = np.zeros([states.shape[0],n_noise,n_hidden])
    data_noise = stim_noise*np.random.randn(n_noise,states.shape[0],n_in)
    for ii in range(n_noise): 
        s_noise[:,ii,:] = sim.run_trial(data_noise[ii,:,:],t_connectivity=False)[1].reshape([states.shape[0],n_hidden])
        
    x_noise,y_noise,mask = build_noise_trials(params)
    mup,mdown,choice,resp = white_noise_test(sim,x_noise)
    coh_out = coherence_test(sim,np.arange(-.2,.2,.01))
    
    sess.close()
