import numpy as np


class Task:
    def __init__(self, task_fun, params):

        self.task_fun = task_fun
        self.dt = params['dt']
        self.tau = params['tau']
        self.alpha = self.dt / self.tau
        self.T = params['T']
        self.N_steps = int(np.ceil(self.T / self.dt))

        self.N_in = params['N_in']
        self.N_out = params['N_out']

        self.stim_noise = params['stim_noise']


    def generate_trial(self, params):

        x_data = np.zeros([self.N_steps, self.N_in])
        y_data = np.zeros([self.N_steps, self.N_out])
        mask = np.zeros([self.N_steps, self.N_out])

        for t in range(self.N_steps):
            x_data[t, :], y_data[t, :], mask[t, :] = self.task_fun(t * self.dt, params)

        x_data += np.sqrt(2 * self.alpha * self.stim_noise * self.stim_noise) * np.random.randn(self.N_steps, self.N_in)

        return x_data, y_data, mask



def generate_batch(list_of_tasks):

    while 1 > 0:

        x_data = []
        y_data = []
        mask = []

        for task in list_of_tasks:
            x,y,m = task.generate_trial()

        yield