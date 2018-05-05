import numpy as np


class Task(object):
    def __init__(self, N_in, N_out, dt, tau, T, N_batch):

        self.N_batch = N_batch
        self.N_in = N_in
        self.N_out = N_out
        self.dt = dt
        self.tau = tau
        self.T = T

        self.alpha = self.dt / self.tau
        self.N_steps = int(np.ceil(self.T / self.dt))

    # return params for a given batch
    def trial_params_function(self, batch, trial):
            pass

    # return input, output, mask, at time t
    def trial_function(self, time, params):
        pass

    def __generate_trial__(self, params):

        x_data = np.zeros([self.N_steps, self.N_in])
        y_data = np.zeros([self.N_steps, self.N_out])
        mask = np.zeros([self.N_steps, self.N_out])

        for t in range(self.N_steps):
            x_data[t, :], y_data[t, :], mask[t, :] = self.trial_function(t * self.dt, params)

        return x_data, y_data, mask

    def batch_generator(self):
        batch = 1
        while batch > 0:

            x_data = []
            y_data = []
            mask = []
            for trial in range(self.N_batch):
                x,y,m = self.__generate_trial__(self.trial_params_function(batch, trial))
                x_data.append(x)
                y_data.append(y)
                mask.append(m)

            batch += 1

            yield np.array(x_data), np.array(y_data), np.array(mask)


class RDM(Task):

    # return params for a given batch
    def trial_params_function(self, batch, trial):

        params = dict()
        params['coherence'] = np.random.choice([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3])
        params['stim_noise'] = 0.1
        params['onset_time'] = np.random.random() * self.T / 2.0
        params['stim_duration'] = np.random.random() * self.T / 4.0 + self.T / 8.0

        return params

    # return input, output, mask, at time t
    def trial_function(self, time, params):

        x_t = np.zeros(self.N_in) + np.sqrt(2*self.alpha*params['stim_noise']*
                                        params['stim_noise'])*np.random.randn(self.N_in)
        y_t = .1 * np.ones(self.N_out)


        coh = params['coherence']
        onset = params['onset_time']
        stim_dur = params['stim_duration']



        if onset < time < onset + stim_dur:
            x_t[0] += coh
            x_t[1] += -coh

        if time > onset + stim_dur:
            y_t[int(coh < 0)] = 1.

        mask_t = np.ones(self.N_out)
        if time < onset + stim_dur:
            mask_t = np.zeros(self.N_out)

        return x_t, y_t, mask_t

