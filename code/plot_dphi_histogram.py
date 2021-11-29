import numpy as np 
import os 
import matplotlib.pyplot as plt 
import argparse
import json

INPUT_DIR = './data/active-matter-5'

# parser = argparse.ArgumentParser()
# parser.add_argument('-s', '--sim_id', metavar='s', type=int, nargs=1, help='the unique \
#     id of the simulation to plot', default=[SIM_ID], required=False)

# id = parser.parse_args().sim_id[0]

params_path = os.path.join(INPUT_DIR, 'params.json')
params = json.loads(open(params_path, 'r').readline())
id = params['id']
data = os.path.join (INPUT_DIR, 'motion_data.csv')
x, y, t, dphi, state = np.transpose(np.loadtxt(data, delimiter=','))

clean_dphi = np.array([i[0] for i in np.transpose([dphi, state]) if i[1] == 0]) \
                * 360 / (2 * np.pi)


binwidth = 10
plt.hist(clean_dphi, bins=np.arange(min(clean_dphi), max(clean_dphi) + binwidth, binwidth))
plt.title("Tumble phase angle difference")
plt.xlabel('Delta phi (degrees)')
plt.savefig(os.path.join(INPUT_DIR, f"dphi_histogram_id={id}"))