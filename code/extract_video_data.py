from os import path, listdir, mkdir
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
from src.animate import make_centroid_animation
from src.imutils import get_frame_centroids
import json 


VIDEO = './videos/10-22-2021/test.MOV'
OUTPUT_DIR = './data/'
MAKE_ANIMATION = True
ANIMATION_TYPE = False 

print('Obtaining centroids...')
x, y, t = get_frame_centroids(VIDEO, end_frame=100)
zeros = np.zeros(len(x))

print('Checking existing data...')
dir_name = "active-matter-"

id = 0
while path.isdir(path.join(OUTPUT_DIR, dir_name + str(id))):
    id += 1

sim_dir = path.join(OUTPUT_DIR, dir_name + str(id))
mkdir(sim_dir)

print(f"Writing motion_data.csv to {sim_dir}...")
motion_data_path = path.join(sim_dir, 'motion_data.csv')
np.savetxt(motion_data_path,
            np.transpose(np.append([x, y, t], [zeros, zeros], axis=0)),
            delimiter=',',
            header='x,y,t,dphi,state')
    
# Save video params in a .json file
print(f"Writing params.json to {sim_dir}...")
params_json_path = path.join(sim_dir, 'params.json')
all_params = {
    "id": id,
    "type": "exp",
    "video": VIDEO,
    "world":
        {
            "dt": (1 / 29.97)
        }
    }

params_json = open(params_json_path, 'w')
params_json.write(json.dumps(all_params))
params_json.close()

ims = sorted([path.join(path.join(path.dirname(VIDEO), 'frames/raw'), img) for 
    img in listdir(path.join(path.dirname(VIDEO), 'frames/raw')) 
    if img.endswith('.jpg')])
    
ims = [Image.open(img) for img in ims]

if MAKE_ANIMATION: 
    # Generating animation
    print("Generating animation...")
    make_centroid_animation(ims, np.transpose([x, y]), 
        sim_dir,
        'overlaid_centroid.avi')

print("Done!")