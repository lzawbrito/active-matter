from src.swimmer import SimpleSwimmer
from src.world import World 
import os.path
import numpy as np 
import json 


# Input parameters
DURATION = 0.03
DT = 0.0001
SIMS_PER_VELOCITY = 4
velocities = np.arange(0, 160e-6, 40e-6)
sim_id = 0
file_dir = "data/sims/varying_v/"

# Instantiate world and swimmer
file_dict = {}
for v in velocities:
    file_sublist = []
    for i in range(SIMS_PER_VELOCITY + 1):
        world = World(DT)
        swimmer = SimpleSwimmer(0, world, v_0=v)

        print(f"Simulating {i}/{SIMS_PER_VELOCITY} for v={v}...")
        positions = []
        t = []
        while swimmer.get_t() <= DURATION:
            positions.append(swimmer.get_position())
            t.append(swimmer.get_t())
            swimmer.step()

        print('Checking existing simulations...')
        file_name = "sim_"
        file_pre = file_dir + file_name 
        file_params = f"_v={round(v*10e6)}_n={i}"
        file_ext = ".csv"
        params_post = '_params.txt'

        # Generate unique simulation ID 
        while os.path.isfile(file_pre + str(sim_id) + file_params + file_ext):
            sim_id += 1
        

        print(f"Writing to {file_pre + str(sim_id) + file_params + file_ext}...")
        data = np.transpose(np.append(np.transpose(positions), [t], axis=0))
        np.savetxt(file_pre + str(sim_id) + file_params + file_ext, data, delimiter=',', 
                header='t,x,y')

        # Save simulation parameters in a .txt file. 
        params = open(file_pre + str(sim_id) + file_params + params_post, 'w')
        params.write(world.params())
        params.write('\n')
        params.write(swimmer.params())
        file_sublist.append(file_pre + str(sim_id) + file_params)

        print(f"""{file_name + str(sim_id) + file_ext} and {file_name + str(sim_id) 
            + params_post} written in {file_dir}.""")
    prop_swimmer = SimpleSwimmer(0, world, v_0=v)
    file_dict["{:.2e}".format(v)] = {"files": file_sublist, "pe": prop_swimmer.pe}

metadata_file = open(file_dir + "metadata_" + str(sim_id) + ".json", 'w')
metadata_file.write(json.dumps(file_dict))

