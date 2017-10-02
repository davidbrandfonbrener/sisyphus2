import numpy as np
import itertools
from sisyphus2.backend.simulation_tools import Simulator
from sisyphus2.tasks import flip_flop
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.animation


def plot_states(states, I):
    if I == []:
        pca = PCA(n_components=4)
        pca.fit(states[:, 0, :])
        reduced = pca.transform(states[:, 0, :])[:, :2]

        fig, ax = plt.subplots()
        a, b = [], []
        sc = ax.scatter(a, b)
        plt.xlim(1.1 * min(reduced[:, 0]), 1.1 * max(reduced[:, 0]))
        plt.ylim(1.1 * min(reduced[:, 1]), 1.1 * max(reduced[:, 1]))

        def animate(i):
            a.append(reduced[i, 0])
            b.append(reduced[i, 1])
            sc.set_offsets(np.c_[a, b])

        ani = matplotlib.animation.FuncAnimation(fig, animate,
                                                 frames=len(reduced[:, 0]), interval=100, repeat=False)
        plt.show()

    else:
        pca = PCA(n_components=4)
        pca.fit(states[:, 0, :])
        reduced = pca.transform(states[:, 0, :])[:, :2]

        I_pca = pca.transform(I)

        fig, ax = plt.subplots()
        a, b = [I_pca[0,0]], [I_pca[0,1]]
        sc = ax.scatter(a, b)
        plt.xlim(1.1 * min(reduced[:, 0]), 1.1 * max(reduced[:, 0]))
        plt.ylim(1.1 * min(reduced[:, 1]), 1.1 * max(reduced[:, 1]))

        def animate(i):
            a.append(reduced[i, 0])
            b.append(reduced[i, 1])
            sc.set_offsets(np.c_[a, b])
            sc.set_color(['r'] + [str(.99 * float(x) / len(reduced[:, 0])) for x in range(len(a))])

        ani = matplotlib.animation.FuncAnimation(fig, animate,
                                                 frames=len(reduced[:, 0]), interval=100, repeat=False)
        plt.show()
    return

# input: simulator object, params dict
# output: fixed points found by hahnloser strategy
#     assumes no biases as currently written
def hahnloser_fixed_point(sim, trial):

    outputs, states = sim.run_trial(trial)

    #TODO decide what input to use
    input_vec = np.matmul(np.absolute(sim.W_in), np.array([.1]*sim.N_in))
    input_mat = np.matmul(np.absolute(sim.W_in), np.transpose(trial))

    #set identity matrix of proper size
    identity = np.diag(np.ones(sim.N_rec))

    points = []
    for i, s in enumerate(states):
        # define active weight matrix
        Wp = np.matmul(np.absolute(sim.W_rec), sim.dale_rec)
        for index in range(sim.N_rec):
            if s[0,index] < 0:
                Wp[:, index] = 0

        # check for fixed point
        I = np.matmul(np.linalg.inv(identity - Wp), input_mat[:, i])

        fixed = True
        for index in range(sim.N_rec):
            if s[0,index] < 0 and I[index] >= 0:
                fixed = False
            if s[0,index] >= 0 and I[index] < 0:
                fixed = False

        if fixed == True:
            # here I is the fixed point, while s is the attained state with the correct sign permutation
            print trial[i, :] #, I, s, pca.transform(I)
            points.append(I)

    return points


if __name__ == '__main__':

    params = flip_flop.set_params(N_batch= 64, N_rec=30,
                           input_wait=5, stim_dur=5, quiet_gap=200, N_turns=4,
                           rec_noise=0, stim_noise=0.0,
                           dale_ratio=.8, tau=20, dt=10.)

    x,y,mask = flip_flop.build_train_batch(params)

    trial = x[0]
    #plt.ylim(-.1, 1.1)
    #plt.plot(trial)
    #plt.show()

    sim = Simulator(params, weights_path="../tasks/weights/flipflop.npz")

    points = hahnloser_fixed_point(sim, trial)

    outputs, states = sim.run_trial(trial)

    plot_states(states=states, I=points)
