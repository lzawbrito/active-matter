from random import gauss
from autograd.core import deprecated_defgrad
import numpy as np
from numpy.linalg import norm

# TODO 
"""
- Right now initialization of SimpleSwimmer solves without accounting for 
  other swimmers that will be in the handler. Think about proper way to handle 
  this.
- Add option to turn off noise term
- Work out on paper what the reflections for a box then compare with sim
"""

class SimpleSwimmer:
    def __init__(self,
                   id, 
                   world, 
                   r = 3,
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
        self.xdot = v_0 * np.cos(phi_0)
        self.ydot = v_0 * np.sin(phi_0)

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
        v = norm([self.xdot, self.ydot])

        # Solve finite difference equations
        phi_i = self.phi + self.omega * dt + np.sqrt(2 * self.d_r * dt) * wphi_i
        x_i = self.x + v * np.cos(self.phi) * dt \
              + self.brown * np.sqrt(2 * self.d_t * dt) * wx_i
        y_i = self.y + v * np.cos(self.phi) * dt \
              + self.brown * np.sqrt(2 * self.d_t * dt) * wy_i
        
        self.phi = phi_i
        self.next_x = x_i
        self.next_y = y_i


class Swimmer:
    def __init__(self, 
                 id,
                 world,
                 height=10, 
                 width=10, 
                 x_0=0, 
                 y_0=0, 
                 v_0=0,
                 phi_0=0, 
                 omega=0):
        """
        Microswimmer object. Default spawn is at the origin with no initial 
        angular or linear velocity, with height and width of 10 pixels.
        Defaults to a non-chiral swimmer.
        """
        self.id = id
        self.world = world
        self.height = height
        self.width = width
        self.x = x_0
        self.y = y_0
        self.v = v_0
        self.phi = phi_0
        self.xdot = v_0 * np.cos(phi_0)
        self.ydot = v_0 * np.sin(phi_0)
        self.omega = omega
        self.d_t = (world.k_B * world.temp) / \
                   (3 * world.viscosity * height * width)
        self.d_r = (world.k_B * world.temp) / \
                   (4 * world.viscosity * height * width)
        # TODO check the above diffusion coefficient correct for rectangle

    
    # def collision(self, handler):
    #     for k in handler.keys():
    #         if k == str(self.id):
    #             continue 
    #         pass


    def step(self, handler: dict):
        """
        Solve for next timestep of differential equation.
        """
        dt = self.world.dt
        gauss_norm = lambda: gauss(0, 1)
        wphi_i = gauss_norm()
        wx_i = gauss_norm()
        wy_i = gauss_norm()
        v = norm([self.xdot, self.ydot])

        # Solve finite difference equations
        phi_i = self.phi + self.omega * dt + np.sqrt(2 * self.d_r * dt) * wphi_i
        x_i = self.x + v * np.cos(self.phi) * dt \
              + np.sqrt(2 * self.d_t * dt) * wx_i
        y_i = self.y + v * np.cos(self.phi) * dt \
              + np.sqrt(2 * self.d_t * dt) * wy_i
        
        self.phi = phi_i
        self.x = x_i
        self.y = y_i
        