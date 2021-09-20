from matplotlib.pyplot import plot
import numpy as np
from os.path import join
from src.plot import plot_swimmer_trajectory

SIM_ID = 1
OUTPUT_DIR = 'plots/trajectories/'

data = f"./data/sims/sim_{SIM_ID}.csv"

x, y, t = np.transpose(np.loadtxt(data, delimiter=','))

positions = np.transpose([x, y])

plot_swimmer_trajectory(positions,
                        join(OUTPUT_DIR, f"swimmer_trajectory_id={SIM_ID}"),
                        title=f"Active Brownian Particle Trajectory",
                        t_start='0', t_stop=round(t[-1], 3))
