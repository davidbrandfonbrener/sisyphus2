import numpy as np
from backend.simulation import Simulator
import flip_flop_old
import matplotlib.pyplot as plt


params = flip_flop_old.set_params(N_batch= 64,
                                  input_wait=5, stim_dur=10, quiet_gap=20, N_turns=2,
                                  rec_noise=0, stim_noise=0,
                                  dale_ratio=.8, tau=100, dt=10.)

x,y,mask = flip_flop_old.build_train_batch(params)

sim = Simulator(params, weights_path="./weights/flipflop.npz")

trial = x[0]
target = y[0]

plt.plot(range(x.shape[1]), trial[:,0], range(x.shape[1]), trial[:,1], range(x.shape[1]), target)
plt.ylim([-.2, 1.2])
plt.show()

outputs, states = sim.run_trial(trial)

plt.plot(range(x.shape[1]), target, range(x.shape[1]), outputs[:,0])
plt.ylim([-.2, 1.2])
plt.show()