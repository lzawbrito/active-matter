import matplotlib.pyplot as plt 
import numpy as np
import json 
from os import path 

FILE_PATH = './data/off-center-perp'

sim_data = json.load(open(path.join(FILE_PATH, 'motion_data.json'), 'r'))
motion_data = {}
for k in sim_data['swimmers'].keys(): 
	s = sim_data['swimmers'][k]
	phi = np.transpose(s['phi'])
	app_om = np.gradient(phi)
	torque = np.gradient(app_om)
	motion_data[k] = {'torque': torque, 'app_om': app_om}

i = 0 
for k in motion_data.keys(): 
	plt.plot(sim_data['t'], motion_data[k]['torque'], label=f'torque_{k}')
	plt.plot(sim_data['t'], motion_data[k]['app_om'], label=f'app_om_{k}')
	i = i + 0.001

plt.legend(loc='best')
plt.title(r'Torque and Apparent $\omega $ (staggered)')
plt.savefig(path.join(FILE_PATH, 'torque_app_om.png'))
	