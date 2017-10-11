import numpy as np
import tensorflow as tf
from backend.networks import Model
import backend.visualizations as V

# Mante task, as in pycog

# Builds a dictionary of parameters that specifies the information
# about an instance of this specific task
def set_params(Name = "mante", N_rec = 150,
               fixation = 20, stimulus = 40, decision=15,
               var_fix_length = 0, var_stim_length = 0,
               stim_noise = 0.1, rec_noise = 0.1,
               dale_ratio=0.8, dt = 10, tau = 100,
               N_batch=64):

    params = dict()

    params['Name'] = Name
    params['N_in'] = 6
    params['N_rec'] = N_rec
    params['N_out'] = 2
    params['N_steps'] = fixation + stimulus + var_fix_length + var_stim_length + decision
    params['N_batch'] = N_batch
    params['stim_noise']       = stim_noise
    params['rec_noise']        = rec_noise
    params['dale_ratio']       = dale_ratio
    params['tau']               = tau
    params['dt']                = dt
    params['alpha']             = dt/tau

    params['fixation']          = fixation
    params['stimulus']          = stimulus
    params['decision']          = decision
    params['var_stim_length']   = var_stim_length
    params['stim_noise']        = stim_noise
    params['rec_noise']         = rec_noise

    return params


def scale(coherence, SCALE=5):
    return (1 + SCALE*coherence/100)/2


def build_train_batch(params):
    N_in     = params['N_in']
    N_out    = params['N_out']
    N_batch = params['N_batch']
    N_steps = params['N_steps']
    fixation = params['fixation']
    stimulus = params['stimulus']
    decision = params['decision']
    var_fix_length = params['var_stim_length']
    var_stim_length = params['var_stim_length']
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

    input_times = np.zeros([N_batch], dtype=np.int)
    output_times = np.zeros([N_batch], dtype=np.int)
    for sample in np.arange(N_batch):
        input_times[sample]  = fixation + fix_delay[sample]
        output_times[sample] = fixation + fix_delay[sample] + stimulus + stim_delay[sample]
    params['input_times'] = input_times
    params['output_times'] = output_times

    contexts = ['m', 'c']
    coherences = [1, 3, 10]
    left_rights = [1, -1]

    x_train = np.zeros([N_batch, N_steps, N_in])
    y_train = 0.5 * np.ones([N_batch, N_steps, N_out])
    mask = np.zeros((N_batch, N_steps, N_out))
    for sample in np.arange(N_batch):
        context = np.random.choice(contexts)
        coh_m = np.random.choice(coherences)
        coh_c = np.random.choice(coherences)
        left_right_m = np.random.choice(left_rights)
        left_right_c = np.random.choice(left_rights)
        input_time = input_times[sample]
        output_time = output_times[sample]
        end_time = output_times[sample] + decision

        if context == 'm':
            left_right = left_right_m
            x_train[sample, input_times[sample]:output_times[sample], 0] = 1.0
        else:
            left_right = left_right_c
            x_train[sample, input_times[sample]:output_times[sample], 1] = 1.0

        if left_right_m > 0: choice_m = 0
        else:                choice_m = 1
        x_train[sample, input_time:output_time, 2+choice_m]     = scale(+coh_m)
        x_train[sample, input_time:output_time, 2+(1-choice_m)] = scale(-coh_m)

        if left_right_c > 0: choice_c = 0
        else:                choice_c = 1
        x_train[sample, input_time:output_time, 4+choice_c]     = scale(+coh_c)
        x_train[sample, input_time:output_time, 4+(1-choice_c)] = scale(-coh_c)

        if left_right > 0: choice = 0
        else:              choice = 1

        y_train[sample, :input_times[sample], :] = lo
        y_train[sample, output_times[sample]:, choice] = hi
        y_train[sample, output_times[sample]:, 1-choice] = lo

        mask[sample, :input_time, :] = 1.0
        mask[sample, output_time:end_time, :] = 1.0

    return x_train, y_train, mask


def generate_train_trials(params):
    while 1 > 0:
        yield build_train_batch(params)

params = set_params()

generator = generate_train_trials(params)
model = Model(params)

configuration = tf.ConfigProto(inter_op_parallelism_threads=10, intra_op_parallelism_threads=10)
sess = tf.Session(config=configuration)
model.train(sess, generator, training_iters=80000)

V.show_W_rec(model, sess)
