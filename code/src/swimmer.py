import numpy as np
from numpy.linalg import norm
from .mathutils import level_curve_intersection, uniform_angle
from numpy.random import normal, uniform
from abc import ABC, abstractmethod, abstractproperty
import json 

class Swimmer(ABC): 
    def dphi(self):
        return self._next_phi - self._phi

    def dr(self): 
        return np.subtract(self.next_position, self.position)

    def solve(self):
        pass 

    def step(self):
        self._x = self._next_x
        self._y = self._next_y
        self._phi = self._next_phi
        self._t += self.world.dt
        self.solve()

    def collide(self):
        tried_r = np.array((self._next_x, self._next_y))
        current_r = np.array((self._x, self._y))

        if self.world.bdy is not None and \
           self.world.is_outside_bdy(tried_r) and \
           self.world.is_inside_bdy(current_r):
            while self.world.bdy(tried_r) > self.world.bdy_pot:
                # TODO maybe abstract this 
                bdy_collision_position = level_curve_intersection((self._x, self._y),
                                                                (self._next_x, 
                                                                    self._next_y),
                                                                self.world.bdy,
                                                                self.world.bdy_pot)

                normal = self.world.normal_vec(bdy_collision_position)
                tried_r = np.subtract(tried_r, 
                                    2 * (np.dot(np.subtract(tried_r, 
                                                            bdy_collision_position),
                                                normal))
                                    * normal)
                current_r = bdy_collision_position 
            self._next_x, self._next_y = tried_r

    @abstractmethod
    def get_state(self):
        pass

class SimpleSwimmer(Swimmer):
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
        self._x = x_0
        self._y = y_0
        self._next_x = x_0
        self._next_y = y_0

        # Velocity 
        self.v = v_0 

        # Angular orientation of particle, angular velocity 
        # TODO add next_phi
        self._phi = phi_0
        self._next_phi = phi_0
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

        # State is constanat 
        self._state = "n/a"

        self._t = 0
        # Solve for next time step upon initialization to determine next_x, 
        # next_y
        self.solve()

    def get_t(self):
        """
        Returns current time of swimmer.
        """
        return self._t


    def get_position(self):
        """
        Return x, y coordinates of swimmer.
        """
        return self._x, self._y

    def get_state(self):
        return self._state

    # def step(self):
        # """
        # Update x and y coordinates and solve for next timestep of differential
        # equation.
        # """
        # self._x = self._next_x
        # self._y = self._next_y
        # self._phi = self._next_phi
        # self._t += self.world.dt
        # self.solve()

    def solve(self):
        """
        Solve for next timestep of differential equation.
        """
        dt = self.world.dt
        wphi_i = self.rot_dist()
        wx_i = self.trans_dist()
        wy_i = self.trans_dist()

        # Solve finite difference equations
        phi_i = self._phi + self.omega * dt + self.rot_diff * np.sqrt(2 * self.d_r * dt) * wphi_i
        x_i = self._x + self.v * np.cos(self._phi) * dt \
              + self.trans_diff * np.sqrt(2 * self.d_t * dt) * wx_i
        y_i = self._y + self.v * np.sin(self._phi) * dt \
              + self.trans_diff * np.sqrt(2 * self.d_t * dt) * wy_i
        
        self._next_phi = phi_i
        self._next_x = x_i
        self._next_y = y_i
        self.collide()

    def params(self):
        params = {
            'R': self.r,
            'v': self.v,
            'omega': self.omega,
            'trans. diffusion': bool(self.trans_diff),
            'trans. diffusion dist.': self.trans_dist.__name__,
            'rot. diffusion': bool(self.rot_diff),
            'rot. diffusion dist.': self.rot_dist.__name__,
            'D_T': self.d_t,
            'D_R': self.d_r,
            'Peclet number': self.pe,
            'duration': self._t
        }
        return params

    def string_params(self):
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
            + f"duration\t\t\t{self._t}\n" 


class RunTumbleSwimmer(Swimmer): 
    def __init__(
            self,
            id, 
            world, 
            t_time, # TODO maybe set defaults for these 
            r_time,
            v=0, # TODO consider changing default v 
            phi=0, 
            r = 0.5e-6,
            x_0 = 0,
            y_0 = 0,
            init_state = 'r',
            tumble_dist = uniform_angle,
            trans_diff=True,
            rot_diff=True, 
            trans_dist=normal,
            rot_dist=uniform_angle):

        if init_state != 'r' and init_state != 't':
            raise ValueError(f"Invalid initial state: \"{init_state}\"" + 
                             " must be one of \"t\" or \"r\".")

        self.id = id 
        self.world = world 
        self.r = r 
        self.t_time = t_time  
        self.r_time = r_time 
        self.v = v
        self._phi = phi
        self._next_phi = phi
        self._x = x_0 
        self._y = y_0 
        self._next_x = x_0
        self._next_y = y_0
        self.running = True if init_state == 'r' else False
        self.time_running = 0
        self.time_tumbling = 0 
        self.tumble_dist = tumble_dist
        self._t = 0
        # Diffusion coefficients
        self.d_t = (world.k_B * world.temp) / \
                        (6 * world.viscosity * self.r)
        self.d_r = (world.k_B * world.temp) / \
                    (8 * world.viscosity * (self.r ** 3))

        self.trans_diff = int(trans_diff)
        self.rot_diff = int(rot_diff)
        self.trans_dist = trans_dist 
        self.rot_dist = rot_dist
        self.pe = self.v / (self.d_r * self.r)
        self.solve()


    def runs_to_time(self, runs):
        return runs * (self.t_time + self.r_time)

    def get_state(self):
        if self.running:
            return 'r'
        else: 
            return 't'

    def solve(self): 
        dt = self.world.dt
        wphi_i = self.rot_dist()
        wx_i = self.trans_dist()
        wy_i = self.trans_dist()
        
        # Update time spent running if running, otherwise update time spent
        # tumbling. If running and time has gone over r_time, change to tumbling 
        # state. If rumbling and time has gone over t_time change to running 
        # state.
        if self.running: 
            self.time_running += dt 
            if self.time_running >= self.r_time: 
                self.running = False 
                self.time_running = 0 
        else: 
            self.time_tumbling += dt
            if self.time_tumbling >= self.t_time: 
                self.running = True 
                self.time_running = 0 
        
        if self.running: 
            self._next_x = self._x + self.v * np.cos(self._phi) \
              + self.trans_diff * np.sqrt(2 * self.d_t * dt) * wx_i
            self._next_y = self._y + self.v * np.sin(self._phi) \
              + self.trans_diff * np.sqrt(2 * self.d_t * dt) * wy_i
        else: 
            self._next_phi = self._phi + self.tumble_dist() \
                + self.rot_diff * np.sqrt(2 * self.d_r * dt) * wphi_i

        self.collide()


    def get_position(self):
        return self._x, self._y

    def get_phi(self):
        return self._phi
        

    def get_t(self): 
        return self._t

    def params(self):
        params = {
            'R': self.r,
            'v': self.v,
            'tau_R': self.r_time,
            'tau_T': self.t_time,
            'tumble_dist': self.tumble_dist.__name__,
            'trans. diffusion': bool(self.trans_diff),
            'trans. diffusion dist.': self.trans_dist.__name__,
            'rot. diffusion': bool(self.rot_diff),
            'rot. diffusion dist.': self.rot_dist.__name__,
            'D_T': self.d_t,
            'D_R': self.d_r,
            'Peclet number': self.pe,
            'duration': self._t
        }
        return params

    def string_params(self):
        return f"RUN AND TUMBLE SWIMMER PARAMETERS:\n" \
             + f"R\t\t\t\t{self.r}\n" \
             + f"v\t\t\t\t{self.v}\n" \
             + f"tau_R\t\t\t\t{self.r_time}\n" \
             + f"tau_T\t\t\t\t{self.t_time}\n" \
             + f"tumble_dist\t\t\t{self.tumble_dist.__name__}\n" \
             + f"trans. diffusion\t\t{bool(self.trans_diff)}\n" \
             + f"trans. diffusion dist.\t\t{self.trans_dist.__name__}\n" \
             + f"rot. diffusion\t\t\t{bool(self.rot_diff)}\n" \
             + f"rot. diffusion dist.\t\t{self.rot_dist.__name__}\n" \
             + f"D_T\t\t\t\t{self.d_t}\n" \
             + f"D_R\t\t\t\t{self.d_r}\n" \
             + f"Peclet number\t\t\t{self.pe}\n" \
             + f"duration\t\t\t{self._t}\n" 

       