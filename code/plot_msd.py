from src.mathutils import msd
import numpy as np 
import matplotlib.pyplot as plt 
from src.plot import plot_swimmer_trajectory
import argparse


SIM_ID = 5
OUTPUT_DIR = 'plots/msd/'
REF_LINE_CONSTANT = 0.5e-11

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sim_id', metavar='s', type=int, nargs=1, help='the unique \
    id of the simulation to plot', default=[SIM_ID], required=False)

id = parser.parse_args().sim_id[0]

data = f"./data/sims/sim_{id}.csv"

x, y, t = np.transpose(np.loadtxt(data, delimiter=','))

print('computing MSDs')
msds = []
for delta_t in t: 
    msds.append(msd(x, y, t[1] - t[0], delta_t=delta_t))

fig, ax = plt.subplots()

plt.loglog(t, msds)
plt.loglog(t, REF_LINE_CONSTANT * t)

ax.set_ylabel("log(msd)")
plt.savefig(OUTPUT_DIR + f"msd_vs_lag_sim={SIM_ID}")
ax.set_xlabel("log(lag)")

