import matplotlib.pyplot as plt 
import numpy as np
import json 
from os import path 

FILE_PATH = './data/'

sim_data = json.load(open(path.join(FILE_PATH, 'motion_data.json'), 'r'))
motion_data = {}
for k in sim_data['swimmers'].keys(): 
	s = sim_data['swimmers'][k]
	x, y = np.transpose(s['pos'])
	app_v_x = np.gradient(x)
	app_v_y = np.gradient(y)

	f_x = np.gradient(app_v_x)
	f_y = np.gradient(app_v_y)

	app_v = np.sqrt(app_v_x **2 + app_v_y **2)
	f = np.sqrt(f_x ** 2 + f_y **2)
	motion_data[k] = {'f': f, 'app_v': app_v}


i = 0
for k in motion_data.keys(): 
	plt.plot(sim_data['t'], motion_data[k]['f'] + i, label=f'f_{k}/m_{k}')
	plt.plot(sim_data['t'], motion_data[k]['app_v'] + i, label=f'app_v_{k}')
	i = i + 0.001


plt.legend(loc='best')
plt.title(r'Force and Apparent Velocity (staggered)')
plt.savefig(path.join(FILE_PATH, 'force_app_v.png'))
	