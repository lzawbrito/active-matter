from autograd import grad
from scipy.constants import k 
from numpy import linalg
import numpy as np 
from math import isclose
import json


class World:
    def __init__(self, dt, bdy=None, bdy_pot=1, temp=310, k_B=k, viscosity=1e-3):
        """
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
        self.n_swimmers = 0
        self.swimmers = {}
        self.interaction_matrix = np.array([])
        self._t = 0

    def add_swimmer(self, swimmer):
        index = self.n_swimmers
        self.swimmers[swimmer.id] = {'swimmer': swimmer, 'index': index}
        self.n_swimmers += 1
        self.make_interaction_matrix()

    def make_interaction_matrix(self):
        self.interaction_matrix = np.zeros((self.n_swimmers, self.n_swimmers, 3))
        self.update_interaction_matrix()

    def update_interaction_matrix(self):
        # Make copy of swimmer dictionary for safety
        d = self.swimmers
        for key_i in d:
            for key_j in d: 
                if d[key_i]['index'] == d[key_j]['index']:
                    continue 
                other = d[key_i]['swimmer']
                potential_term = d[key_j]['swimmer'].potential(other) 
                i, j = d[key_i]['index'], d[key_j]['index']
                self.interaction_matrix[i, j] = potential_term

    def get_t(self): 
        return self._t


    def get_interaction_term(self, id): 
        """
        Returns
        -------
        2D Array representing x and y components of the potential at that 
        location. 
        """
        row = self.swimmers[id]['index']
        return np.sum(self.interaction_matrix[row], axis=0)
        


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

    def step(self):
        self._t += self.dt
        for swimmer in self.swimmers.keys(): 
            self.swimmers[swimmer]['swimmer'].step()

        self.update_interaction_matrix()


if __name__ == '__main__':
    world = World(0.1, lambda x: x[0] ** 2 + x[1] ** 2)
    # print(world.normal_vec((1, 1)))
    # print(world.normal_vec(1, 1))
    assert isclose(linalg.norm(world.normal_vec((1, 1))), 1, abs_tol=1e-6)
    assert isclose(linalg.norm(world.normal_vec(1, 1)), 1, abs_tol=1e-6)
    


        