from src.animate import make_swimmer_animation
from src.world import World 
from src.swimmer import RectangleSwimmer
import numpy as np
import matplotlib.pyplot as plt 
import os 
import json 
from src.boundaries import rectangular_bdy
from random import random 

DURATION = 0.3
DT = 0.01
OUTPUT_DIR = './data'
STIFFNESS = 1 # beta
COMPRESS = 0.01 # gamma 
INTER_STRENGTH = 0.07 # phi
N_SWIMMERS = 50  
H = 3 
W = 1

i = 1 
world = World(DT, bdy=rectangular_bdy, bdy_pot=25)
while i <= N_SWIMMERS: 
	x = random() * 50 - 25 
	y = random() * 50 - 25 
	phi = random() * 2 * np.pi
	RectangleSwimmer(str(i), world, STIFFNESS, COMPRESS, INTER_STRENGTH, h=H, w=W,
					 x_0=y, y_0=x, v_0=0, phi_0=phi, 
					 rot_diff=False, trans_diff=False)
	i += 1


swimmer_data = {}
sim_data = {'t': []}  

for k in world.swimmers.keys():
	swimmer_data[k] = {
		'pos': [],
		'phi': [], 
		'h': world.swimmers[k]['swimmer'].h, 
		'w': world.swimmers[k]['swimmer'].w
	}

steps = 0

print('Simulating...')
while world.get_t() <= DURATION: 
	sim_data['t'].append(world.get_t())
	for k in world.swimmers.keys():
		s = world.swimmers[k]['swimmer']
		swimmer_data[k]['pos'].append(s.get_position())
		swimmer_data[k]['phi'].append(s.get_phi())
	steps += 1 
	world.step()

sim_data['swimmers'] = swimmer_data

print('Checking existing simulations...')
dir_name = "n-body-"

id = 0
while os.path.isdir(os.path.join(OUTPUT_DIR, dir_name + str(id))):
    id += 1

sim_dir = os.path.join(OUTPUT_DIR, dir_name + str(id))
os.mkdir(sim_dir)
motion_data_path = os.path.join(sim_dir, 'motion_data.json')
print(f"Writing motion_data.csv to {sim_dir}...")
f = open(motion_data_path, 'w')
json.dump(sim_data, f)
f.close()

print('Making animation...')
make_swimmer_animation(swimmer_data, sim_data['t'], sim_dir, 'animation.avi',
					   xlim=(-50, 50), ylim=(-50, 50))


print(f"Writing params.txt to {sim_dir}...")
# Save simulation parameters in a .txt file. 
params_txt_path = os.path.join(sim_dir, 'params.txt')
params = open(params_txt_path, 'w')
params.write(world.string_params())
params.write('\n')
for k in world.swimmers.keys(): 
	params.write(world.swimmers[k]['swimmer'].string_params())
	params.write('\n')
params.close()

print(f"Writing params.json to {sim_dir}...")
# Save simulation params in a .json file
params_json_path = os.path.join(sim_dir, 'params.json')
world_params_dict = world.params()
all_params = {
    "id": id,
    "type": "sim",
    "world": world_params_dict, 
	"swimmers": {},
    "sim": {
        }
    }
for k in world.swimmers.keys(): 
	all_params['swimmers'][k] = world.swimmers[k]['swimmer'].params()
params_json = open(params_json_path, 'w')
params_json.write(json.dumps(all_params))
params_json.close()
print('Done!')

