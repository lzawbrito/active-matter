from math import isclose
from random import random
import autograd.numpy as np
import autograd 
from autograd import grad 

# TODO 
"""
- if two intersections chose one closest to r1. Figure out how to do that. 
  maybe something with the random point that is chosen. Right now it 
  converges on a random intersection. 
- tolerance is a bit iffy. how quickly the algorithm converges depends 
  on the slope of the function near the level curve. so it is hard to 
  predict how close the algorithm's output will be to the actual intersection 
  point. Ideally want greater slopes near the level curve.
- figure out better way to determine whether there is an intersection?
"""


def msd(x, y, dt, delta_t):
    """
    Calculates mean squared displacement (MSD) of the given positions x, y using 
    time intervals delta_t. Because time steps are discretized, delta_t rounds 
    to nearest integer.
    """
    interval = int(round(delta_t / dt))
    assert len(x) == len(y)

    i = 0
    sum = 0
    while i + interval < len(x):
        r_1 = np.array((x[i], y[i]))
        r_2 = np.array((x[i + interval], y[i + interval]))
        dr = np.subtract(r_2, r_1)
        norm_sq = np.linalg.norm(dr) ** 2
        sum += norm_sq
        i += 1
    try:
        return sum / i
    except ZeroDivisionError:
        print('delta_t too large.')
    

def level_curve_intersection(r1, r2, bdy, potential, tol=1e-10, max_iter=100): 
    """
    Determines the intersection of the line between r1 and r2 and the given 
    level curve. 
    """
    r1, r2 = np.array(r1), np.array(r2)
    dr = np.subtract(r1, r2)
    x = np.add(random() * dr, r1)
    grad_bdy = grad(bdy)

    i = 0
    while not isclose(bdy(x), potential, abs_tol=tol) and i <= max_iter:
        normal = grad_bdy((float(x[0]), float(x[1])))
        dr_orientation = np.sign(np.dot(dr, normal))
        if bdy(x) < potential: 
            if dr_orientation > 0:
                r1 = x 
            else:  
                r2 = x 
        elif bdy(x) > potential: 
            if dr_orientation > 0: 
                r2 = x 
            else:  
                r1 = x 
        else:      
            return x 
        dr = np.subtract(r1, r2)
        x = np.add(random() * dr, r1)
        grad_bdy = grad(bdy)
        i += 1
    if i <= max_iter:
        return x 
    else:
        return None 


if __name__ == '__main__':
    assert isclose(msd([1, 2, 3], [0, 0, 0], 0.1, 0.1), 1, abs_tol=1e-8)
    assert isclose(msd([1, 2, 3, 4], [0, 0, 0, 0], 0.1, 0.2), 4, abs_tol=1e-8)

    assert isclose(level_curve_intersection((1, 2), (2, 3),
                                            lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2),
                                            3)[0],
                                            -0.5 + np.sqrt(68) / 4,
                                            abs_tol=1e-8) and \
           isclose(level_curve_intersection((1, 2), (2, 3),
                                            lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2),
                                            3)[1],
                                            0.5 + np.sqrt(68) / 4,
                                            abs_tol=1e-8)

    assert isclose(level_curve_intersection((-2, -1), (-3, -2),
                                            lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2),
                                            3)[0],
                                            -0.5 - np.sqrt(68) / 4,
                                            abs_tol=1e-8) and \
           isclose(level_curve_intersection((-2, -1), (-3, -2),
                                            lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2),
                                            3)[1],
                                            0.5 - np.sqrt(68) / 4,
                                            abs_tol=1e-8)

    def ripple(x): 
        return np.exp(- 100 * (np.sqrt(x[0] ** 2 + x[1] ** 2) - 3) ** 2)

    assert isclose(level_curve_intersection((-2, -1), (-3, -2),
                                            ripple,
                                            1)[0],
                                            - 0.5 - np.sqrt(68) / 4,
                                            abs_tol=1e-5) and \
           isclose(level_curve_intersection((-2, -1), (-3, -2),
                                            ripple,
                                            1)[1],
                                            0.5 - np.sqrt(68) / 4,
                                            abs_tol=1e-5)

