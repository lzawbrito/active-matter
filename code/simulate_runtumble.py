from src.swimmer import RunTumbleSwimmer
from src.world import World 
import os.path
import numpy as np 
import json 

N_RUNS = 75
DT = 0.02
V = 7e-7

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
file_dir = "data/sims/"
file_name = "sim_"
file_pre = file_dir + file_name
file_ext = ".csv"

sim_id = 0
while os.path.isfile(file_pre + str(sim_id) + file_ext):
    sim_id += 1

print(f"Writing to {file_pre + str(sim_id) + file_ext}...")
data = np.transpose(np.append(np.transpose(positions), [t, dphi, state], axis=0))
np.savetxt(file_pre + str(sim_id) + file_ext, data, delimiter=',', 
           header='x,y,t,dphi,state')

# Save simulation parameters in a .txt file. 
params = open(file_pre + str(sim_id) + '_params.txt', 'w')
params.write(world.string_params())
params.write('\n')
params.write(swimmer.string_params())
params.write('\n')
params.write(f"SIM PARAMETERS\nruns\t\t\t\t{N_RUNS}")
params.close()

swimmer_params_dict = swimmer.params()
world_params_dict = world.params()
all_params = {
    "world": world_params_dict, 
    "swimmer": swimmer_params_dict,
    "sim": {
        "runs": N_RUNS
        }
    }

params_json = open(file_pre + str(sim_id) + '_params.json', 'w')
params_json.write(json.dumps(all_params))
params_json.close()

print(f"""{file_name + str(sim_id) + file_ext}, {file_name + str(sim_id) +
'_params.json'} and {file_name + str(sim_id) + '_params.txt'} written in
{file_dir}.""")