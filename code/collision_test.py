import matplotlib.pyplot as plt
from src.mathutils import level_curve_intersection
from src.world import World
import numpy as np
from src.plot import plot_swimmer_trajectory
from src.boundaries import rectangular_bdy


bdy = rectangular_bdy #lambda x: x[0]
bdy_pot = 2
world = World(1, bdy=bdy, bdy_pot=bdy_pot)

x = 1.75 
y = 1.20
v_x, v_y = 1, 1
positions = [(x, y)]
for i in range(0, 1):
    next_x = x + v_x 
    next_y = y + v_y 
    tried_r = np.array((next_x, next_y))
    current_r = np.array((x, y))
    if bdy is not None and \
        bdy(tried_r) > bdy_pot and \
        bdy(current_r) < bdy_pot:
        while bdy(tried_r) > bdy_pot:
            bdy_collision_position = level_curve_intersection((x, y),
                                                            (next_x, 
                                                                next_y),
                                                            bdy,
                                                            bdy_pot)

            normal = world.normal_vec(bdy_collision_position)
            tried_r = np.subtract(tried_r, 
                                2 * (np.dot(np.subtract(tried_r, 
                                                        bdy_collision_position),
                                            normal))
                                * normal)
            current_r = bdy_collision_position 
            positions.append(bdy_collision_position)
        next_x, next_y = tried_r
    x = next_x
    y = next_y
    positions.append((next_x, next_y))

print(positions)
plot_swimmer_trajectory(positions, 'plots/collision_test')