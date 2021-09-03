# TODO 
"""
- import scipy for boltzmann constant 
- find gradient operator usage in scipy/numpy
"""

class World:
    def __init__(self, dt, bdy=lambda x, y: 0, bdy_constant=1, temp=300, k_B=1, viscosity=1):
        """`
        World object: this is where one defines constants, 
        timestep sizes, etc.

        Boundary: this is a function that takes x, y coordinates as input and 
        returns a Boolean. The function should return False if the x, y 
        coordinates are outside the boundary and True otherwise. 
        """
        self.temp = temp
        self.dt = dt
        self.bdy
        self.k_B = k_B
        self.viscosity = viscosity

        