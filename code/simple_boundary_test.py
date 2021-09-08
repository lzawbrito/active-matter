from src.swimmer import SimpleSwimmer
from src.world import World 
import numpy as np 
from src.plot import plot_swimmer_trajectory
from src.boundaries import circular_bdy, rectangular_bdy

# TODO implement function that returns lambda of rectangle function but with 
#      length and width as arguments.
# perhaps what is happening is the reflection algorithm puts the swimmer 
# outside the boundary. fix this by iteratively checking to make sure we 
# are not placing it outside. 


DURATION = 30
DT = 0.1

# world = World(0.01, bdy=rectangular_bdy, bdy_pot=1e-6)
# swimmer = SimpleSwimmer(0, world, x_0=0.9e-6, y_0=0.96e-6, v_0=1e-7, phi_0=np.pi/3, brown=False)

world = World(0.01, bdy=circular)

positions = []
for i in range(0, int(DURATION / DT)):
    positions.append(swimmer.get_position())
    swimmer.step()

plot_swimmer_trajectory(positions, "plots/rectangular_bdy")

x, y = np.transpose(positions)

