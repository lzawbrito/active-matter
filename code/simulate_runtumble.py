from src.swimmer import RunTumbleSwimmer
from src.world import World 
import os.path
import numpy as np 
import json 

N_RUNS = 75
DT = 0.02
V = 7e-7
OUTPUT_DIR = './data'

world = World(DT)
swimmer = RunTumbleSwimmer(0, world, t_time=0.01, r_time=1, v=V, trans_diff=False, rot_diff=False, r=5e-7)
duration = swimmer.runs_to_time(N_RUNS)

print('Simulating...')
positions = []
t = []
dphi = []
state = []
while swimmer.get_t() <= duration:
    positions.append(swimmer.get_position())
    t.append(swimmer.get_t())
    dphi.append(swimmer.dphi())

    # Running:  1 
    # Tumbling: 0
    if swimmer.get_state() == 'r':
        state.append(1)
    else: 
        state.append(0)
    swimmer.step()

print('Checking existing simulations...')
dir_name = "active-matter-"

id = 0
while os.path.isdir(os.path.join(OUTPUT_DIR, dir_name + str(id))):
    id += 1

sim_dir = os.path.join(OUTPUT_DIR, dir_name + str(id))
os.mkdir(sim_dir)
motion_data_path = os.path.join(sim_dir, 'motion_data.csv')
print(f"Writing motion_data.csv to {sim_dir}...")
data = np.transpose(np.append(np.transpose(positions), [t, dphi, state], axis=0))
np.savetxt(motion_data_path, data, delimiter=',', 
           header='x,y,t,dphi,state')

print(f"Writing params.txt to {sim_dir}...")
# Save simulation parameters in a .txt file. 
params_txt_path = os.path.join(sim_dir, 'params.txt')
params = open(params_txt_path, 'w')
params.write(world.string_params())
params.write('\n')
params.write(swimmer.string_params())
params.write('\n')
params.write(f"SIM PARAMETERS\nruns\t\t\t\t{N_RUNS}")
params.close()

print(f"Writing params.json to {sim_dir}...")
# Save simulation params in a .json file
params_json_path = os.path.join(sim_dir, 'params.json')
swimmer_params_dict = swimmer.params()
world_params_dict = world.params()
all_params = {
    "id": id,
    "type": "sim",
    "world": world_params_dict, 
    "swimmer": swimmer_params_dict,
    "sim": {
        "runs": N_RUNS
        }
    }

params_json = open(params_json_path, 'w')
params_json.write(json.dumps(all_params))
params_json.close()
