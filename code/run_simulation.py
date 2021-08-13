from src.swimmer import Swimmer 
from src.world import World 
import random 

"""
TODO 
- make toy version of this that runs in console, just for fun
- Figure out where we will put API for calculating MSD, etc.
"""

DT = 0.01

world = World(DT)
handler = {}
for i in range(0, 100):
    handler[str(i)] = Swimmer(str(i),
                              world, 
                              x_0 = 10 * random.random(), 
                              y_0 = 10 * random.random())


