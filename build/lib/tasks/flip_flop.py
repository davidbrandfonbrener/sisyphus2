import numpy as np
import tensorflow as tf
from backend import networks
import backend.visualizations as V
#from backend.simulation_tools import Simulator

# flip_flop task

# Builds a dictionary of parameters that specifies the information
# about an instance of this specific task
def set_params(Name = "flip_flop", N_rec = 50,
               N_turns = 3, input_wait = 3, quiet_gap = 4, stim_dur = 3,
               var_delay_length = 0, stim_noise = 0.1, rec_noise = .1,
               N_batch = 128, dale_ratio=0.8, dt = 10, tau = 100,
               biases = False, seed=None):

    params = dict()

    params['Name'] = Name
    params['N_in'] = 2
    params['N_rec'] = N_rec
    params['N_out'] = 2
    params['N_steps'] = input_wait + N_turns * (stim_dur + quiet_gap + var_delay_length)
    params['N_batch'] = N_batch
    params['stim_noise']       = stim_noise
    params['rec_noise']        = rec_noise
    params['dale_ratio']       = dale_ratio
    params['tau']               = tau
        #[tau] * int(N_rec * dale_ratio) + [2 * tau] * int(N_rec * (1 - dale_ratio))
    params['dt']                = dt
    params['alpha']             = dt * 1.0/tau

    params['N_turns']          = N_turns
    params['input_wait']       = input_wait
    params['quiet_gap']        = quiet_gap
    params['stim_dur']         = stim_dur
    params['var_delay_length'] = var_delay_length

    params['biases'] = biases

    params['input_connectivity_mask'] = None

    connect = np.ones((N_rec, N_rec))
    for i in range(N_rec):
        connect[i,i] = 0
    params['recurrent_connectivity_mask'] = connect
    params['output_connectivity_mask'] = None


    params['L1_in'] = 0
    params['L1_rec'] = 0
    params['L1_out'] = 0
    params['L2_in'] = 0 #.1
    params['L2_rec'] = 0
    params['L2_out'] = 0 #.1
    params['L2_firing_rate'] = 0.1
    params['sussillo_constant'] = 0.001

    return params



# This generates the training data for our network
# It will be a set of input_times and output_times for when we expect input
# and when the corresponding output is expected
def build_train_batch(params):
    N_in = params['N_in']
    N_out = params['N_out']
    N_batch = params['N_batch']
    N_steps = params['N_steps']
    N_turns = params['N_turns']
    input_wait = params['input_wait']
    quiet_gap = params['quiet_gap']
    stim_dur = params['stim_dur']
    var_delay_length = params['var_delay_length']
    stim_noise = params['stim_noise']

    if var_delay_length == 0:
        var_delay = np.zeros(N_batch, dtype=int)
    else:
        var_delay = np.random.randint(var_delay_length, size=N_batch) + 1

    input_times  = np.zeros([N_batch, N_turns], dtype=np.int)
    output_times = np.zeros([N_batch, N_turns], dtype=np.int)

    turn_time = np.zeros(N_batch, dtype=np.int)

    for sample in np.arange(N_batch):
        turn_time[sample] = stim_dur + quiet_gap + var_delay[sample]
        for i in np.arange(N_turns):
            input_times[sample, i] = input_wait + i * turn_time[sample]
            output_times[sample, i] = input_wait + i * turn_time[sample] + stim_dur

    x_train = np.ones([N_batch, N_steps, N_in]) * .1
    y_train = np.ones([N_batch, N_steps, N_out]) * .1
    mask = np.zeros((N_batch, N_steps, N_out))
    for sample in np.arange(N_batch):
        for turn in np.arange(N_turns):
            firing_neuron = np.random.randint(2)  # 0 or 1
            x_train[sample, input_times[sample, turn]:(input_times[sample, turn] + stim_dur), firing_neuron] = 1.0
            y_train[sample, output_times[sample, turn]:(input_times[sample, turn] + turn_time[sample]), firing_neuron] = 1.0
        mask[sample, :, 0] = [0.0 if (x[0] == .1 and x[1] == .1) else 1.0 for x in y_train[sample]]


    x_train = x_train + stim_noise * np.random.randn(N_batch, N_steps, 2)
    params['input_times'] = input_times
    params['output_times'] = output_times
    return x_train, y_train, mask


def generate_train_trials(params):
    while 1 > 0:
        yield build_train_batch(params)



if __name__ == '__main__':

    params = set_params(N_batch= 64, N_rec=30,
                        input_wait=5, stim_dur=10, quiet_gap=10, N_turns=3,
                        rec_noise=0.01, stim_noise=0.01, biases=False, var_delay_length=40,
                        dale_ratio=.8, tau=20, dt=10.)

    generator = generate_train_trials(params)
    model = networks.Model(params)

    configuration = tf.ConfigProto(inter_op_parallelism_threads=10, intra_op_parallelism_threads=10)
    sess = tf.Session(config=configuration)
    model.train(sess, generator, training_iters=10000, learning_rate=.001, weights_path="./weights/flipflop.npz")

    data = generator.next()
    V.visualize_2_input_one_output_trial(model, sess, data)

    #V.show_W_in(model, sess)
    V.show_W_rec(model, sess)
    #V.show_W_out(model, sess)

    #sim = Simulator(params, weights_path="./weights/flipflop.npz")
    #sim.run_trials(data[0], 100)
