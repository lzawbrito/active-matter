import numpy as np
from numpy.linalg import norm
from .mathutils import level_curve_intersection
from numpy.random import normal, uniform

# TODO 
"""
- Right now initialization of SimpleSwimmer solves without accounting for 
  other swimmers that will be in the handler. Think about proper way to handle 
  this.
- MSD 
- simulate simple scenarios mimicking paper 
    - input 1 micron diameter 10-3 viscosity water, T = 310 K, calculate D_R 
    - set D_t = 0 
    - initial conditions: x_0 = 0, y_0 = 0, phi_0 = 0, 
    - v = 0 - brownian (turn translational back on)
        = 10 micron/s
    - Pe = v/\sqrt{D_T D_R}
    - suggested = Pe = v/(RD_R)
    - Pe = t_diffusion / t_advection
- at end of simulation, x, y, t columns of an array. MSD: mean squared 
  displacement. assume delta t is one start with t = 0, 
"""
def uniform_angle():
    """
    Draws samples from a random distribution [0, 2 * pi]
    """
    return uniform(low=0.0, high=2 * np.pi)


class SimpleSwimmer:
    def __init__(self,
                   id, 
                   world, 
                   r = 0.5e-6,
                   x_0 = 0,
                   y_0 = 0,
                   v_0 = 0,
                   phi_0 = 0,
                   omega = 0,
                   trans_diff=True,
                   rot_diff=True, 
                   trans_dist=normal,
                   rot_dist=uniform_angle):
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

        self.trans_diff = int(trans_diff)
        self.rot_diff = int(rot_diff)
        self.trans_dist = trans_dist 
        self.rot_dist = rot_dist
        
        # Peclet number 
        self.pe = self.v / (self.d_r * self.r)

        self.t = 0
        # Solve for next time step upon initialization to determine next_x, 
        # next_y
        self.solve()

    def get_t(self):
        """
        Returns current time of swimmer.
        """
        return self.t


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
        self.t += self.world.dt
        self.solve()

    def solve(self):
        """
        Solve for next timestep of differential equation.
        """
        dt = self.world.dt
        wphi_i = self.rot_dist()
        wx_i = self.trans_dist()
        wy_i = self.trans_dist()

        # Solve finite difference equations
        phi_i = self.phi + self.omega * dt + self.rot_diff * np.sqrt(2 * self.d_r * dt) * wphi_i
        x_i = self.x + self.v * np.cos(self.phi) * dt \
              + self.trans_diff * np.sqrt(2 * self.d_t * dt) * wx_i
        y_i = self.y + self.v * np.sin(self.phi) * dt \
              + self.trans_diff * np.sqrt(2 * self.d_t * dt) * wy_i
        
        self.phi = phi_i
        self.next_x = x_i
        self.next_y = y_i
        tried_r = np.array((self.next_x, self.next_y))
        current_r = np.array((self.x, self.y))
        if self.world.bdy is not None and \
           self.world.is_outside_bdy(tried_r) and \
           self.world.is_inside_bdy(current_r):
            while self.world.bdy(tried_r) > self.world.bdy_pot:
                # TODO maybe abstract this 
                bdy_collision_position = level_curve_intersection((self.x, self.y),
                                                                (self.next_x, 
                                                                    self.next_y),
                                                                self.world.bdy,
                                                                self.world.bdy_pot)

                normal = self.world.normal_vec(bdy_collision_position)
                tried_r = np.subtract(tried_r, 
                                    2 * (np.dot(np.subtract(tried_r, 
                                                            bdy_collision_position),
                                                normal))
                                    * normal)
                current_r = bdy_collision_position 
            self.next_x, self.next_y = tried_r

    def params(self):
        return f"SWIMMER PARAMETERS:\n" \
            + f"R\t\t\t\t{self.r}\n" \
            + f"v\t\t\t\t{self.v}\n" \
            + f"omega\t\t\t\t{self.omega}\n" \
            + f"trans. diffusion\t\t{bool(self.trans_diff)}\n" \
            + f"rot. diffusion\t\t\t{bool(self.rot_diff)}\n" \
            + f"trans. diffusion dist.\t\t{self.trans_dist.__name__}\n" \
            + f"rot. diffusion dist.\t\t{self.rot_dist.__name__}\n" \
            + f"D_T\t\t\t\t{self.d_t}\n" \
            + f"D_R\t\t\t\t{self.d_r}\n" \
            + f"Peclet number\t\t\t{self.pe}\n" \
            + f"duration\t\t\t{self.t}\n" 

