from src.animate import make_swimmer_animation
from src.world import World 
from src.swimmer import RectangleSwimmer
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection
from src.mathutils import pos_angle2vertices

DURATION = 2
DT = 0.01
OUTPUT_DIR = './data'
STIFFNESS = 1
COMPRESS = 0.01
INTER_STRENGTH = 0.1
H = 3 
W = 1

world = World(DT)
s1 = RectangleSwimmer('1', world, STIFFNESS, COMPRESS, INTER_STRENGTH, h=H, w=W,
							x_0=3, v_0=4, phi_0=(3 * np.pi) / 4 + 0.2)

s2 = RectangleSwimmer('2', world, STIFFNESS, COMPRESS, INTER_STRENGTH, h=H, w=W,
							x_0=-3, v_0=4, phi_0=np.pi / 4 - 0.2)

# s3 = RectangleSwimmer('3', world, STIFFNESS, COMPRESS, INTER_STRENGTH, h=H, w=W,
# 							y_0=3, v_0=2, phi_0=(3 * np.pi) / 2)

# world.make_interaction_matrix()

swimmer_data = {}

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
	world.step()
	for k in world.swimmers.keys():
		s = world.swimmers[k]['swimmer']
		swimmer_data[k]['pos'].append(s.get_position())
		swimmer_data[k]['phi'].append(s.get_phi())
	steps += 1 


print('Making animation...')
make_swimmer_animation(swimmer_data, steps, './test', 'collision.avi')


print(s1.string_params())
print('Done!')

