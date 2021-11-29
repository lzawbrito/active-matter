from matplotlib.pyplot import plot
import numpy as np
from os.path import join
from src.plot import plot_swimmer_trajectory
import argparse

INPUT_DIR = './data/active-matter-0'

# parser = argparse.ArgumentParser()
# parser.add_argument('-s', '--sim_i0d', metavar='s', type=int, nargs=1, help='the unique \
#     id of the simulation to plot', default=[SIM_ID], required=False)

# id = parser.parse_args().sim_id[0]
id = 'exp'
data = join(INPUT_DIR, 'motion_data.csv')

x, y, t, dphi, state = np.transpose(np.loadtxt(data, delimiter=','))

positions = np.transpose([x, y])

plot_swimmer_trajectory(positions,
                        join(INPUT_DIR, f"swimmer_trajectory_id={id}"),
                        title=f"Active Brownian Particle Trajectory",
                        t_start='0', t_stop=round(t[-1], 3))
