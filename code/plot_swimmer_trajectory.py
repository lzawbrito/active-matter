from matplotlib.pyplot import plot
from src.swimmer import SimpleSwimmer
from src.world import World 
import numpy as np
from os.path import join
from src.plot import plot_swimmer_trajectory

DT = 0.01
DURATION = 1000
OUTPUT_DIR = 'plots'

world = World(DT)
swimmer = SimpleSwimmer(0, world)

positions = []
for i in range(0, int(DURATION / DT)):
    positions.append(swimmer.get_position())
    swimmer.step()

x, y = np.transpose(positions)

plot_swimmer_trajectory(positions,
                        join(OUTPUT_DIR, f"swimmer_trajectory_t={DURATION}_v={swimmer.v}"),
                        swimmer, world,
                        title=f"Active Brownian Particle Trajectory",
                        t_start='0', t_stop=DURATION)
