from matplotlib.pyplot import plot
import numpy as np
from os.path import join
from src.plot import plot_swimmer_trajectory
import argparse

DATA_DIR = 'data/videos/10-22-2021'
OUTPUT_DIR = 'plots/trajectories/'

# TODO take directory instead
# parser = argparse.ArgumentParser()
# parser.add_argument('-s', '--sim_id', metavar='s', type=int, nargs=1, help='the unique \
#     id of the simulation to plot', default=[SIM_ID], required=False)

# id = parser.parse_args().sim_id[0]
id = 'exp'
data = join(DATA_DIR, 'data.csv')

x, y, t, dphi, state = np.loadtxt(data, delimiter=',')

positions = np.transpose([x, y])

plot_swimmer_trajectory(positions,
                        join(OUTPUT_DIR, f"swimmer_trajectory_id={id}"),
                        title=f"Active Brownian Particle Trajectory",
                        t_start='0', t_stop=round(t[-1], 3))
