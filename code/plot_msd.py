from src.mathutils import msd, msds_for_lag
import numpy as np 
import os.path
import matplotlib.pyplot as plt 
import argparse
import json


SIM_ID = 8
OUTPUT_DIR = 'plots/msd/'
REF_LINE_CONSTANT = 0.20e-10
REF_LINE_CONSTANT2 = 0.40e-10
RUNTUMBLE_NORM = True

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sim_id', metavar='s', type=int, nargs=1, help='the unique \
    id of the simulation to plot', default=[SIM_ID], required=False)

id = parser.parse_args().sim_id[0]
data = f"./data/sims/sim_{id}.csv"
params_path = f"./data/sims/sim_{id}_params.json"
params = json.loads(open(params_path, 'r').readline())

norm = 1
if RUNTUMBLE_NORM:
    tau_r = params['swimmer']['tau_R']
    tau_t = params['swimmer']['tau_T']
    norm = tau_r + tau_t

dt = params['world']['dt']

x, y, t, dphi, state = np.transpose(np.loadtxt(data, delimiter=','))

filename = f"./data/sims/msd_vs_lag_data_sim={SIM_ID}.csv"
msds = []
if os.path.isfile(filename):
    msds = np.loadtxt(filename, delimiter=',')
else: 
    print('computing MSDs')
    for delta_t in t: 
        msds.append(msd(x, y, t[1] - t[0], delta_t=delta_t))
    np.savetxt(filename, np.transpose(msds), delimiter=',')

fig, ax = plt.subplots()

plt.loglog(t / norm, msds)

# Plot reference lines: 
# Log-log plot is log(y) = k log(t) + log(a) => Y(t) = mX + log(a). We want to 
# solve for a when X = 0. X = log(t) = 0 => t = 1. Thus we want 
# log(a) = log(y(t = 1)) => a = y(t = 1), or, in this case, a = msd(1).
# timestep * dt = t, so setting t = 1 we find timestep = 1 / dt.
plt.loglog(t / norm, msds[int(1 / dt)] * t) 
plt.loglog(t / norm, msds[int(1 / dt)] * t ** 2, label="slope 2")
xlabel = "log(lag) (tau_r + tau_t)" if RUNTUMBLE_NORM else "log(lag) (s)"
plt.xlabel(xlabel)
ax.set_ylabel("log(msd)")
plt.savefig(OUTPUT_DIR + f"msd_vs_lag_sim={SIM_ID}")

