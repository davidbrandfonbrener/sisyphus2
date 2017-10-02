import numpy as np


class Task(object):
    """ Abstract class for cognitive tasks.
    Meant to be implemented by the task in this file

    """
    default_params = None

    def __init__(self, *params, **kwargs):
        for key in self.default_params:
            setattr(self, key, self.default_params[key])
        for dictionary in params:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def build_train_batch(self):
        pass

    def generate_train_trials(self):
        while 1 > 0:
            yield self.build_train_batch()




class rdm(Task):

    default_params = dict(n_in = 1, n_out = 1, n_steps = 200, coherences=[.5], stim_noise = 0, rec_noise = 0,
                            L1_rec = 0, L2_firing_rate = 0, batch_size = 128,
                            sample_size = 128, epochs = 100, N_rec = 50, dale_ratio=0.8,
                            tau=100.0, dt = 10.0, biases = True,
                            task='n_back', rt_version=False)

    def build_train_batch(self):

        input_times = np.zeros([self.batch_size, self.n_in], dtype=np.int)
        output_times = np.zeros([self.batch_size, self.n_out], dtype=np.int)

        x_train = np.zeros([self.batch_size, self.n_steps, self.n_in])
        y_train = np.zeros([self.batch_size, self.n_steps, self.n_out])
        mask = np.ones((self.batch_size, self.n_steps, self.n_in))

        stim_time = range(40, 140)
        if self.rt_version:
            out_time = range(50, 200)
        else:
            out_time = range(160, 200)

        dirs = np.random.choice([-1, 1], replace=True, size=(self.batch_size))
        cohs = np.random.choice(self.coherences, replace=True, size=(self.batch_size))
        stims = dirs * cohs;

        for ii in range(self.batch_size):
            x_train[ii, stim_time, 0] = stims[ii]
            y_train[ii, out_time, 0] = dirs[ii]

        x_train = x_train + self.stim_noise * np.random.randn(self.batch_size, self.n_steps, self.n_in)
        self.input_times = input_times
        self.output_times = output_times

        return x_train, y_train, mask






class flip_flop(Task):
    default_params = dict(Name = "flip_flop", N_rec = 50, N_in = 2, N_out = 2,
               N_turns = 3, input_wait = 3, quiet_gap = 4, stim_dur = 3,
               var_delay_length = 0, stim_noise = 0.1, rec_noise = .1,
               N_batch = 128, dale_ratio=0.8, dt = 10, tau = 100,
               biases = False, seed=None)

    def build_train_batch(self):

        setattr(self, 'N_steps',
                self.input_wait + self.N_turns * (self.stim_dur + self.quiet_gap + self.var_delay_length))

        if self.var_delay_length == 0:
            var_delay = np.zeros(self.N_batch, dtype=int)
        else:
            var_delay = np.random.randint(self.var_delay_length, size=self.N_batch) + 1

        input_times = np.zeros([self.N_batch, self.N_turns], dtype=np.int)
        output_times = np.zeros([self.N_batch, self.N_turns], dtype=np.int)

        turn_time = np.zeros(self.N_batch, dtype=np.int)

        for sample in np.arange(self.N_batch):
            turn_time[sample] = self.stim_dur + self.quiet_gap + var_delay[sample]
            for i in np.arange(self.N_turns):
                input_times[sample, i] = self.input_wait + i * turn_time[sample]
                output_times[sample, i] = self.input_wait + i * turn_time[sample] + self.stim_dur

        x_train = np.ones([self.N_batch, self.N_steps, self.N_in]) * .1
        y_train = np.ones([self.N_batch, self.N_steps, self.N_out]) * .1
        mask = np.zeros((self.N_batch, self.N_steps, self.N_out))
        for sample in np.arange(self.N_batch):
            for turn in np.arange(self.N_turns):
                firing_neuron = np.random.randint(2)  # 0 or 1
                x_train[sample, input_times[sample, turn]:(input_times[sample, turn] + self.stim_dur), firing_neuron] = 1.0
                y_train[sample, output_times[sample, turn]:(input_times[sample, turn]
                                                            + turn_time[sample]),firing_neuron] = 1.0
            mask[sample, :, 0] = [0.0 if (x[0] == .1 and x[1] == .1) else 1.0 for x in y_train[sample]]

        x_train = x_train + self.stim_noise * np.random.randn(self.N_batch, self.N_steps, self.N_in)
        self.input_times = input_times
        self.output_times = output_times

        return x_train, y_train, mask
