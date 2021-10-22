import numpy as np 
import matplotlib.pyplot as plt 
import argparse

SIM_ID = 4
OUTPUT_DIR = 'plots/dphi-histograms/'

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sim_id', metavar='s', type=int, nargs=1, help='the unique \
    id of the simulation to plot', default=[SIM_ID], required=False)

id = parser.parse_args().sim_id[0]
data = f"./data/sims/sim_{id}.csv"
x, y, t, dphi, state = np.transpose(np.loadtxt(data, delimiter=','))

clean_dphi = np.array([i[0] for i in np.transpose([dphi, state]) if i[1] == 0]) \
                * 360 / (2 * np.pi)


binwidth = 10
plt.hist(clean_dphi, bins=np.arange(min(clean_dphi), max(clean_dphi) + binwidth, binwidth))
plt.title("Tumble phase angle difference")
plt.xlabel('Delta phi (degrees)')
plt.savefig(OUTPUT_DIR + f"dphi_histogram_sim={id}")