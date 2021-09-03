from autograd import grad
from scipy.constants import k 
from numpy import linalg
import numpy as np 
from math import isclose


class World:
    def __init__(self, dt, bdy=lambda x: 0, bdy_constant=1, temp=300, k_B=k, viscosity=1):
        """`
        World object: this is where one defines constants, 
        timestep sizes, etc.

        Boundary: this is a function that takes x, y coordinates as input and 
        returns a Boolean. The function should return False if the x, y 
        coordinates are outside the boundary and True otherwise. 
        """
        self.temp = temp
        self.dt = dt
        self.bdy = bdy 
        self.k_B = k_B
        self.viscosity = viscosity

    def normal_vec(self, x, y=None):
        grad_bdy = grad(self.bdy)
        sol = None 
        if y is not None:
            sol = grad_bdy((float(x), float(y)))
        else:
            x_float = []
            for i in x:
                x_float.append(float(i))
            sol = grad_bdy(x_float)
            # TODO make sure it points inward...?

        norm = linalg.norm(sol)
        return np.divide(sol, norm)



##################### TEST SUITE #####################
if __name__ == '__main__':
    world = World(0.1, lambda x: x[0] ** 2 + x[1] ** 2)
    # print(world.normal_vec((1, 1)))
    # print(world.normal_vec(1, 1))
    assert isclose(linalg.norm(world.normal_vec((1, 1))), 1, abs_tol=1e-6)
    assert isclose(linalg.norm(world.normal_vec(1, 1)), 1, abs_tol=1e-6)
    


        