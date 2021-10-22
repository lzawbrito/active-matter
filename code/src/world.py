from autograd import grad
from scipy.constants import k 
from numpy import linalg
import numpy as np 
from math import isclose
import json


class World:
    def __init__(self, dt, bdy=None, bdy_pot=1, temp=310, k_B=k, viscosity=1e-3):
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
        self.bdy_pot = bdy_pot
        self.k_B = k_B
        self.viscosity = viscosity

    def normal_vec(self, x, y=None):
        if self.bdy is None: 
            return None
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
        return -1 * np.divide(sol, norm)
    
    def is_outside_bdy(self, r):
        return self.bdy(r) > self.bdy_pot
        
    def is_inside_bdy(self, r):
        return self.bdy(r) < self.bdy_pot

    def string_params(self):
        bdy_name = self.bdy if self.bdy is not None else 'None'
        return f"WORLD CONDITIONS:\n" \
                + f"temperature\t\t\t{self.temp}\n" \
                + f"k_B\t\t\t\t{self.k_B}\n" \
                + f"viscosity\t\t\t{self.viscosity}\n" \
                + f"boundary\t\t\t{bdy_name}\n" \
                + f"dt\t\t\t\t{self.dt}\n" 

    def params(self): 
        bdy_name = self.bdy if self.bdy is not None else 'None'
        params = {
            'temperature': self.temp,
            'k_B': self.k_B,
            'viscosity': self.viscosity,
            'boundary': str(self.bdy),
            'dt': self.dt
        }
        return params


if __name__ == '__main__':
    world = World(0.1, lambda x: x[0] ** 2 + x[1] ** 2)
    # print(world.normal_vec((1, 1)))
    # print(world.normal_vec(1, 1))
    assert isclose(linalg.norm(world.normal_vec((1, 1))), 1, abs_tol=1e-6)
    assert isclose(linalg.norm(world.normal_vec(1, 1)), 1, abs_tol=1e-6)
    


        