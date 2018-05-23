import numpy as np
import tensorflow as tf
from backend.networks import Model
import backend.visualizations as V

# Romo task, as in pycog

# Builds a dictionary of parameters that specifies the information
# about an instance of this specific task
def set_params(Name = "romo", N_rec = 500,
               fixation = 25, stim_1 = 25, delay = 150, stim_2 = 25, decision=15,
               var_fix_length = 0, var_stim_length = 0,
               stim_noise = 0.1, rec_noise = 0.1,
               dale_ratio=0.8, dt = 20, tau = 100,
               N_batch=64,L2_firing_rate=0.):

    N_steps = fixation + var_fix_length + stim_1 + stim_2 + 2 * var_stim_length + decision + delay + 100
    params = dict()

    params['Name']            = Name
    params['N_rec']           = N_rec
    params['N_in']            = 2
    params['N_out']           = 2
    params['N_steps']         = N_steps
    params['N_batch']         = N_batch
    params['stim_noise']      = stim_noise
    params['rec_noise']       = rec_noise
    params['dale_ratio']      = dale_ratio
    params['tau']             = tau
    params['dt']              = dt
    params['alpha']           = dt/tau

    params['fixation']        = fixation
    params['stim_1']          = stim_1
    params['delay']           = delay
    params['stim_2']          = stim_2
    params['decision']        = decision
    params['var_fix_length']  = var_fix_length
    params['var_stim_length'] = var_stim_length
    params['stim_noise']      = stim_noise
    params['rec_noise']       = rec_noise
    params['L2_firing_rate']  = L2_firing_rate

    return params


def scale_p(f):
    return 0.4 + 0.8*(f - 10)/(34 - 10)


def scale_n(f):
    return 0.4 + 0.8*(34 - f)/(34 - 10)


def build_train_trials(params):
    N_in    = params['N_in']
    N_out   = params['N_out']
    N_batch = params['N_batch']
    N_steps = params['N_steps']
    fixation = params['fixation']
    var_fix_length = params['var_fix_length']
    var_stim_length = params['var_stim_length']
    stim_1 = params['stim_1']
    delay = params['delay']
    stim_2 = params['stim_2']
    lo = 0.2
    hi = 1.0

    if var_stim_length == 0:
        stim_delay = np.zeros(N_batch, dtype=int)
    else:
        stim_delay = np.random.randint(var_stim_length, size=N_batch) + 1
    if var_fix_length == 0:
        fix_delay = np.zeros(N_batch, dtype=int)
    else:
        fix_delay = np.random.randint(var_fix_length, size=N_batch) + 1

    input_times  = np.zeros([N_batch, 2], dtype=np.int)
    output_times = np.zeros([N_batch, 2], dtype=np.int)
    for sample in np.arange(N_batch):
        input_times[sample, 0]  = fixation + fix_delay[sample]
        input_times[sample, 1]  = fixation + fix_delay[sample] + stim_1 + stim_delay[sample] + delay
        
        output_times[sample, 0] = fixation + fix_delay[sample] + stim_1 + stim_delay[sample]
        output_times[sample, 1] = fixation + fix_delay[sample] + stim_1 + 2*stim_delay[sample] + delay + stim_2
    params['input_times'] = input_times
    params['output_times'] = output_times

    fpairs = [(18, 10), (22, 14), (26, 18), (30, 22), (34, 26)]
    gt_lts = ['>', '<']

    x_train = np.zeros([N_batch, N_steps, N_in])
    y_train = 0.5 * np.ones([N_batch, N_steps, N_out])
    mask = np.zeros((N_batch, N_steps, N_out))
    for sample in np.arange(N_batch):
        fpair = fpairs[np.random.choice(len(fpairs))]
        gt_lt = np.random.choice(gt_lts)
        if gt_lt == '>':
            f1, f2 = fpair
            choice = 0
        else:
            f2, f1 = fpair
            choice = 1
            
        buzz_1 = [input_times[sample, 0], input_times[sample, 0] + stim_1 + stim_delay[sample]]
        x_train[sample, buzz_1[0]:buzz_1[1], 0] = scale_p(f1)
        x_train[sample, buzz_1[0]:buzz_1[1], 1] = scale_n(f1)
        
        buzz_2 = [input_times[sample, 1], input_times[sample, 1] + stim_1 + stim_delay[sample]]
        x_train[sample, buzz_2[0]:buzz_2[1], 0] = scale_p(f2)
        x_train[sample, buzz_2[0]:buzz_2[1], 1] = scale_n(f2)

        y_train[sample, :input_times[sample, 0], :] = lo
        y_train[sample, output_times[sample, 1]:, choice] = hi
        y_train[sample, output_times[sample, 1]:, 1-choice] = lo

        mask[sample, :input_times[sample, 0], :]  = 1.0
        mask[sample, output_times[sample, 1]:, :] = 1.0

    return x_train, y_train, mask


def generate_train_trials(params):
    while 1 > 0:
        yield build_train_trials(params)


if __name__ == "__main__":
    params = set_params()

    generator = generate_train_trials(params)

    print "time steps:", params["N_in"]
    print "N_batch:", params["N_batch"]

    model = Model(params)

    configuration = tf.ConfigProto(inter_op_parallelism_threads=10, intra_op_parallelism_threads=10)
    sess = tf.Session(config=configuration)
    model.train(sess, generator, training_iters=10000)

    V.show_W_rec(model, sess)