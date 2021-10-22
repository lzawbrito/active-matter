from matplotlib.pyplot import xlim
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


DURATION = 10
DT = 0.0001
R = 20e-6


world = World(DT, bdy=circular_bdy, bdy_pot=R)
swimmer = SimpleSwimmer(0, world, y_0=0.5e-9, phi_0=np.pi / 3, v_0=10e-6)

print('computing circular boundary trajectory')
positions = []
for i in range(0, int(DURATION / DT)):
    positions.append(swimmer.get_position())
    swimmer.step()

plot_swimmer_trajectory(positions, "plots/circular_bdy", xlim=(-R, R), ylim=(-R, R))

world = World(0.01, bdy=rectangular_bdy, bdy_pot=R)
swimmer = SimpleSwimmer(0, world, y_0=0.5e-9, phi_0=np.pi / 3, v_0=10e-6)

print('computing rectangular boundary trajectory')
positions = []
for i in range(0, int(DURATION / DT)):
    positions.append(swimmer.get_position())
    swimmer.step()

plot_swimmer_trajectory(positions, "plots/rectangular_bdy", xlim=(-R - R / 10, 
                        R + R / 10), ylim=(-R - R / 10, R + R / 10))