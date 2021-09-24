from src.mathutils import msd
import numpy as np 
import matplotlib.pyplot as plt 
import json 


SIM_ID = 0
OUTPUT_DIR = 'plots/msd/'
SIMS_DIR = 'data/sims/varying_v/'
REF_LINE_CONSTANT = 0.5e-11
REF_LINE_CONSTANT2 = 0.5e-7

md = open(SIMS_DIR + 'metadata_' + str(SIM_ID) + '.json', 'r')
unparsed_files = md.read()
files_dict = json.loads(unparsed_files)

fig, ax = plt.subplots()

for key in files_dict: 
    file_set = files_dict[key]['files']
    msds_for_v = []
    for f in file_set: 
        x, y, t = np.transpose(np.loadtxt(f + '.csv', delimiter=','))
        params = open(f + "_params.txt", 'r')
        param_lines = params.readlines()
        peclet = float(files_dict[key]['pe'])
        v = float(key)

        print('computing MSDs for ' + f)
        msds = []
        for delta_t in t: 
            msds.append(msd(x, y, t[1] - t[0], delta_t=delta_t))
        msds_for_v.append(msds)

    avg_msd = np.sum(msds_for_v, axis=0) / len(msds_for_v)

    plt.loglog(t, avg_msd, label=f"v={round(v*10e6, 2)} micron/s, pe={round(peclet, 2)}")

plt.loglog(t, REF_LINE_CONSTANT * t, label="slope 1")
plt.loglog(t, REF_LINE_CONSTANT2 * t ** 2, label="slope 2")
ax.set_ylabel("log(msd)")
ax.set_xlabel("log(lag)")
plt.legend()

plt.savefig(OUTPUT_DIR + f"msd_vs_lag_varying_v_sim={SIM_ID}")
