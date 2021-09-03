from src.swimmer import SimpleSwimmer, Swimmer 
from src.world import World 
import matplotlib.pyplot as plt
import random 
import numpy as np 

DT = 0.01
DURATION = 10

world = World(DT)
swimmer = SimpleSwimmer(0, world)

positions = []
for i in range(0, int(DURATION / DT)):
    positions.append(swimmer.get_position())
    swimmer.step()

trans_positions = np.transpose(positions)
plt.plot(trans_positions[0], trans_positions[1])
plt.gca().set_aspect('equal')
plt.show()