from math import isclose
from random import random
import numpy as np

# TODO 
"""
- test level_curve_intersection, write function signature
"""

def level_curve_intersection(f, level_curve, x_bounds, level_curve_value):
    x = random() * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    while not isclose(level_curve(x, f(x)), level_curve_value, abs_tol=1e-8):
        print(x, f(x))
        if level_curve(x, f(x)) > level_curve_value:
            x_bounds = (x_bounds[0], x)
        elif level_curve(x, f(x)) < level_curve_value:
            x_bounds = (x, x_bounds[1])
        else:
            break 

        x = random() * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    return x, f(x)


def line_intersection(line1, line2):
    """
    From https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


if __name__ == '__main__':
    print(line_intersection(((1, 1), (4, 3)), ((4, 1), (3, 3))))
    level_curve_intersection(lambda x: x + 1, lambda x, y: np.sqrt(x ** 2 + y ** 2), [1, 2], 3)