from src.swimmer import SimpleSwimmer
from src.world import World 
import os.path
import numpy as np 
import json


# Input parameters
DURATION = 0.1
DT = 0.0001
V = 10e-6

# Instantiate world and swimmer
world = World(DT)
swimmer = SimpleSwimmer(0, world, v_0=V)

print('Simulating...')
positions = []
t = []
while swimmer.get_t() <= DURATION:
    positions.append(swimmer.get_position())
    t.append(swimmer.get_t())
    swimmer.step()

print('Checking existing simulations...')
file_dir = "data/sims/"
file_name = "sim_"
file_pre = file_dir + file_name
file_ext = ".csv"
params_post = '_params.txt'

# Generate unique simulation ID 
sim_id = 0
while os.path.isfile(file_pre + str(sim_id) + file_ext):
    sim_id += 1

print(f"Writing to {file_pre + str(sim_id) + file_ext}...")
data = np.transpose(np.append(np.transpose(positions), [t], axis=0))
np.savetxt(file_pre + str(sim_id) + file_ext, data, delimiter=',', 
           header='t,x,y')


# Save simulation parameters in a .txt file. 
params = open(file_pre + str(sim_id) + '_params.txt', 'w')
params.write(world.string_params())
params.write('\n')
params.write(swimmer.string_params())
params.write('\n')
params.close()

swimmer_params_dict = swimmer.params()
world_params_dict = world.params()
all_params = {
    "world": world_params_dict, 
    "swimmer": swimmer_params_dict,
    }

params_json = open(file_pre + str(sim_id) + '_params.json', 'w')
params_json.write(json.dumps(all_params))
params_json.close()

print(f"""{file_name + str(sim_id) + file_ext}, {file_name + str(sim_id) +
'_params.json'} and {file_name + str(sim_id) + '_params.txt'} written in
{file_dir}.""")