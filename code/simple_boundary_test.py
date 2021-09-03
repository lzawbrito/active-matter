from src.swimmer import SimpleSwimmer, Swimmer 
from src.world import World 
import matplotlib.pyplot as plt
import numpy as np 

# TODO implement function that returns lambda of rectangle function but with 
#      length and width as arguments.

def rectangular_bdy(x):
    if x[0] <= x[1] and -x[0] <= x[1]:
        return x[1]
    elif x[0] >= x[1] and -x[0] <= x[1]:
        return x[0]
    elif x[0] > x[1] and -x[0] > x[1]:
        return -x[1] 
    elif x[0] < x[1] and -x[0] > x[1]: 
        return -x[0]
    else:
        return x[0]
    

world = World(0.01, bdy=rectangular_bdy)
print(world.normal_vec((-1, 0)))


