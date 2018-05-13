import numpy as np


class Task(object):
    def __init__(self, N_in, N_out, dt, tau, T, N_batch):

        # ----------------------------------
        # Initialize required parameters
        # ----------------------------------
        self.N_batch = N_batch
        self.N_in = N_in
        self.N_out = N_out
        self.dt = dt
        self.tau = tau
        self.T = T

        # ----------------------------------
        # Calculate implied parameters
        # ----------------------------------
        self.alpha = (1.0 * self.dt) / self.tau
        self.N_steps = int(np.ceil(self.T / self.dt))

    def trial_params_function(self, batch, trial):
            pass

    def trial_function(self, time, params):
        pass

    def generate_trial(self, params):

        # ----------------------------------
        # Loop to generate a single trial
        # ----------------------------------
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
            # ----------------------------------
            # Loop over trials in batch
            # ----------------------------------
            for trial in range(self.N_batch):
                # ---------------------------------------
                # Generate each trial based on its params
                # ---------------------------------------
                x,y,m = self.generate_trial(self.trial_params_function(batch, trial))
                x_data.append(x)
                y_data.append(y)
                mask.append(m)

            batch += 1

            yield np.array(x_data), np.array(y_data), np.array(mask)


class RDM(Task):

    def trial_params_function(self, batch, trial):

        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
        params = dict()
        params['coherence'] = np.random.choice([0.3])
        params['direction'] = np.random.choice([0, 1])
        params['stim_noise'] = 0.1
        params['onset_time'] = np.random.random() * self.T / 2.0
        params['stim_duration'] = np.random.random() * self.T / 4.0 + self.T / 8.0

        return params

    def trial_function(self, time, params):

        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2*self.alpha*params['stim_noise']*params['stim_noise'])*np.random.randn(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)

        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        coh = params['coherence']
        onset = params['onset_time']
        stim_dur = params['stim_duration']
        dir = params['direction']

        # ----------------------------------
        # Compute values
        # ----------------------------------
        if onset < time < onset + stim_dur:
            x_t[dir] += 1 + coh
            x_t[(dir + 1) % 2] += 1

        if time > onset + stim_dur + 20:
            y_t[dir] = 1.

        # if time < onset + stim_dur:
        #    mask_t = np.zeros(self.N_out)

        return x_t, y_t, mask_t

