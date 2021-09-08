from operator import le
from random import gauss
from autograd.core import deprecated_defgrad
import numpy as np
from numpy.linalg import norm
from .mathutils import level_curve_intersection

# TODO 
"""
- Right now initialization of SimpleSwimmer solves without accounting for 
  other swimmers that will be in the handler. Think about proper way to handle 
  this.
- Work out on paper what the reflections for a box then compare with sim
- Boundaries currently have to be monotonically increasing from the interior. 
"""

class SimpleSwimmer:
    def __init__(self,
                   id, 
                   world, 
                   r = 1e-9,
                   x_0 = 0,
                   y_0 = 0,
                   v_0 = 0,
                   phi_0 = 0,
                   omega = 0,
                   brown=True):
        """
        Simple microswimmer object from Volpe, et al.
        """

        # Unique identifier for handling purposes
        self.id = id 

        # World, see src/world
        self.world = world 

        # Extended body's radius 
        self.r = r

        # Positions
        self.x = x_0
        self.y = y_0
        self.next_x = x_0
        self.next_y = y_0

        # Velocity 
        self.v = v_0 

        # Angular orientation of particle, angular velocity 
        self.phi = phi_0
        self.omega = omega

        # Diffusion coefficients
        self.d_t = (world.k_B * world.temp) / \
                        (6 * world.viscosity * self.r)
        self.d_r = (world.k_B * world.temp) / \
                    (8 * world.viscosity * (self.r ** 3))

        self.brown = int(brown)

        # Solve for next time step upon initialization to determine next_x, 
        # next_y
        self.solve()

    def get_position(self):
        """
        Return x, y coordinates of swimmer.
        """
        return self.x, self.y

    def delta_xy(self):
        """
        Computes change in position between current timestep and next. 
        """
        return self.next_x - self.x, self.next_y - self.next_y

    def step(self):
        """
        Update x and y coordinates and solve for next timestep of differential
        equation.
        """
        self.x = self.next_x
        self.y = self.next_y
        self.solve()

    def solve(self):
        """
        Solve for next timestep of differential equation.
        """
        dt = self.world.dt
        gauss_norm = lambda: gauss(0, 1)
        wphi_i = gauss_norm()
        wx_i = gauss_norm()
        wy_i = gauss_norm()
        # Solve finite difference equations
        phi_i = self.phi + self.omega * dt + self.brown * np.sqrt(2 * self.d_r * dt) * wphi_i
        x_i = self.x + self.v * np.cos(self.phi) * dt \
              + self.brown * np.sqrt(2 * self.d_t * dt) * wx_i
        y_i = self.y + self.v * np.sin(self.phi) * dt \
              + self.brown * np.sqrt(2 * self.d_t * dt) * wy_i
        
        self.phi = phi_i
        self.next_x = x_i
        self.next_y = y_i
        tried_r = np.array((self.next_x, self.next_y))
        print(tried_r)
        current_r = np.array((self.x, self.y))
        if self.world.bdy is not None and \
           self.world.bdy(tried_r) > self.world.bdy_pot and \
           self.world.bdy(current_r) < self.world.bdy_pot:
            bdy_collision_position = level_curve_intersection((self.x, self.y),
                                                            (self.next_x, 
                                                                self.next_y),
                                                            self.world.bdy,
                                                            self.world.bdy_pot)

            normal = self.world.normal_vec(bdy_collision_position)
            next_r = np.subtract(tried_r, 
                                 2 * (np.dot(np.subtract(tried_r, 
                                                         bdy_collision_position),
                                             normal))
                                 * normal)
            self.next_x, self.next_y = next_r

