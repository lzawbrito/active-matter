# TODO import scipy for boltzmann constant 

class World:
    def __init__(self, dt, temp=300, k_B=1, viscosity=1):
        """`
        World object: this is where one defines constants, 
        timestep sizes, etc.
        """
        self.temp = temp
        self.dt = dt
        self.k_B = k_B
        self.viscosity = viscosity
        