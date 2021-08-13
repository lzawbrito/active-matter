from random import gauss
import numpy as np


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
        v = np.norm([self.xdot, self.ydot])

        # Solve finite difference equations
        phi_i = self.phi + self.omega * dt + np.sqrt(2 * self.d_r * dt) * wphi_i
        x_i = self.x + v * np.cos(self.phi) * dt \
              + np.sqrt(2 * self.d_t * dt) * wx_i
        y_i = self.y + v * np.cos(self.phi) * dt \
              + np.sqrt(2 * self.d_t * dt) * wy_i
        
        self.phi = phi_i
        self.x = x_i
        self.y = y_i
        