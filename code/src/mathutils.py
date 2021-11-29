from math import isclose
from random import random
import autograd.numpy as np
import autograd 
from autograd import grad
from numpy.random import normal, uniform
from threading import Thread
import concurrent.futures 

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


class NoSolutionError(Exception):
    pass


class InfinitelyManySolutionsError(Exception):
    pass 


def msds_for_lag(x, y, dt, delta_ts):
    assert len(x) == len(y)
    global msds 
    msds = [] 
    threads = [] 
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(delta_ts)) as executor: 
        executor.map(lambda tdelta_t: ((tdelta_t, msd(x, y, dt, tdelta_t))), delta_ts)
    # for delta_t in delta_ts: 
    #     t = Thread(target=lambda tx, ty, tdt, tdelta_t: msds.append((tdelta_t, msd(tx, ty, tdt, tdelta_t))),
    #         args=(x, y, dt, delta_t))
    #     threads.append(t)
    #     t.start()
    msds = [x[1] for x in sorted(msds, key=lambda x: x[0])]

    return msds

def msd(x, y, dt, delta_t):
    """
    Calculates mean squared displacement (MSD) of the given positions x, y using 
    time intervals delta_t. Because time steps are discretized, delta_t rounds 
    to nearest integer.
    """
    assert len(x) == len(y)
    interval = int(round(delta_t / dt))

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


def uniform_angle():
    """
    Draws samples from a random distribution [0, 2 * pi]
    """
    return uniform(low=0.0, high=2 * np.pi)


def line_seg_intersection(a1, a2, b1, b2):
    """
    CONSIDER generalization to n dimensional line segments?
    """
    a1 = np.array(a1)
    a2 = np.array(a2)
    b1 = np.array(b1)
    b2 = np.array(b2)
    da = a2 - a1 
    db = b2 - b1 
    ab = b1 - a1 
    
    daxdb = np.cross(da, db)
    abxda = np.cross(ab, da)

    if daxdb == 0 and abxda == 0:  
        # Collinear
        raise InfinitelyManySolutionsError('Lines are collinear.')
    elif daxdb == 0 and abxda != 0: 
        # Non intersecting and parallel
        raise NoSolutionError('Non-intersecting lines.')

    alpha = np.cross(ab, db) / daxdb
    beta = abxda / daxdb

    if daxdb !=0 and abxda != 0 and 0 <= alpha <= 1 and 0 <= beta <= 1:
        return a1 + alpha * da
    else: 
        # Non-intersecting and non-parallel
        raise NoSolutionError('Non-intersecting lines.')
        

def align_rectangles(a_verts, b_verts):
    """

    Params
    ------
    - a_verts: vertices of rectangle a, given in either clockwise or counter-
               clockwise order. 
    - b_verts: vertices of rectangle b, given in either clockwise or counter-
               clockwise order. 

    Returns
    ------
    
    """
    assert np.shape(a_verts) == (4, 2)
    assert np.shape(b_verts) == (4, 2)

    a0 = np.array(a_verts[0])
    a1 = np.array(a_verts[1])
    a2 = np.array(a_verts[2])

    mid = (1 / 2) * (a2 - a0) + a0
    edge = a1 - a0
    theta = np.pi / 2 if edge[0] == 0 else np.arctan(edge[1] / edge[0])
    new_a0, new_a1, new_a2, new_a3 = [unitary_affine_transform(v, theta, (-1) * mid) for v in a_verts]
    new_b0, new_b1, new_b2, new_b3 = [unitary_affine_transform(v, theta, (-1) * mid) for v in b_verts]

    return [new_a0, new_a1, new_a2, new_a3], [new_b0, new_b1, new_b2, new_b3]


def rectangle_overlap(a_verts, b_verts):
    """

    Params
    ------
    - a_verts: vertices of rectangle a, given in either clockwise or counter-
               clockwise order. 
    - b_verts: vertices of rectangle b, given in either clockwise or counter-
               clockwise order. 

    Returns
    ------
    
    """
    # Align rectangles such that centroid of a is at the origin and its sides 
    # are parallel to the axes of the coordinate system.
    trans_a_verts, trans_b_verts = align_rectangles(a_verts, b_verts)
    a0 = trans_a_verts[0]
    a1 = trans_a_verts[1]
    a2 = trans_a_verts[2]
    a3 = trans_a_verts[3]

    b0 = trans_b_verts[0]
    b1 = trans_b_verts[1]
    b2 = trans_b_verts[2]
    b3 = trans_b_verts[3]
    a_sides = [[a0, a1], [a1, a2], [a2, a3], [a3, a0]]
    b_sides = [[b0, b1], [b1, b2], [b2, b3], [b3, b0]]

    vertices = []

    # Find vertices of polygon by checking intersections between sides of 
    # rectangles. 
    for a_from, a_to in a_sides: 
        for b_from, b_to in b_sides: 
            try:
                vertices.append(line_seg_intersection(a_from, a_to, b_from, b_to))
            except NoSolutionError as e:
                continue 
            except InfinitelyManySolutionsError as e: 
                continue    


    # Find vertices of polygon by checking for vertices of rectangle b that 
    # are in rectangle a 
    
    a_verts_in_b = vertices_inside(trans_a_verts, trans_b_verts)
    b_verts_in_a = vertices_inside(trans_b_verts, trans_a_verts)
    for v in a_verts_in_b:
        vertices.append(v)

    for v in b_verts_in_a:
        vertices.append(v)
    return area(vertices)


def vertices_inside(a_verts, b_verts):
    """
    
    Params
    ------
    - `a_verts`: vertices of outer rectangle
    - `b_verts`: vertices of inner rectangle

    Returns
    ------
    - a list of rectangle b's vertices inside rectangle a in the original 
        coordinate system.
        in 
    """
    assert np.shape(a_verts) == (4, 2)
    assert np.shape(b_verts) == (4, 2)

    a0 = np.array(a_verts[0])
    a1 = np.array(a_verts[1])
    a2 = np.array(a_verts[2])

    mid = (1 / 2) * (a2 - a0) + a0
    edge = a1 - a0
    theta = np.pi / 2 if edge[0] == 0 else np.arctan(edge[1] / edge[0])
    trans_a_verts = [{
        'new': unitary_affine_transform(v, theta, (-1) * mid), 
        'original': v
        } for v in a_verts]
    trans_b_verts = [{
        'new': unitary_affine_transform(v, theta, (-1) * mid), 
        'original': v
        } for v in b_verts]

    a_xlim = np.min([i['new'][0] for i in trans_a_verts]), \
        np.max([i['new'][0] for i in trans_a_verts])
    a_ylim = np.min([i['new'][1] for i in trans_a_verts]), \
        np.max([i['new'][1] for i in trans_a_verts])

    vertices = []

    for b_vert in trans_b_verts:
        if a_xlim[0] < b_vert['new'][0] < a_xlim[1] and a_ylim[0] < b_vert['new'][1] < a_ylim[1]:
            vertices.append(b_vert['original'])

    return vertices 


def unitary_affine_transform(v, theta, dr):
    """
    Transform vector by the given rotation and translation. Translates the 
    vector then rotates it clockwise by theta. 
    """
    v = np.array(v)
    translated = v + np.array(dr)
    return np.matmul(ccw_rotation_matrix(theta), translated)[:2]


def ccw_rotation_matrix(theta):
    """
    Two by two clockwise rotation matrix by angle theta.
    
    Params
    ------
    - `theta`: angle by which to perform rotation 

    Returns 
    -------
    - a 2 by 2 matrix representing this transformation.
    """
    return np.array([[np.cos(theta), np.sin(theta)], 
                     [- np.sin(theta), np.cos(theta)]])


def area(verts):
    """
    The shoelace method is given by 
        A = (1/2) | (Sum_i^{n-1} x_i y_{i+1}) + x_n y_1  
                  - (Sum_i^{n-1} x_{i+1} y_i) - x_1 y_n |

    Params
    ------
    - `verts`: a list of vertices of the polyhedron:
               [(x_1, y_2), ..., (x_n, y_n)] 

    Returns
    ------
    - the area of the given polyhedron
    """
    # Sort vertices for shoelace method 
    if len(verts) == 0: 
        return 0 

    com = (1 / len(verts)) * np.sum(np.transpose(verts)[0]), \
          (1 / len(verts)) * np.sum(np.transpose(verts)[1])
    verts = sorted(verts, key=lambda vert: np.arctan2(vert[1] - com[1], vert[0] - com[0]))

    sum = verts[-1][0] * verts[0][1] - verts[0][0] * verts[-1][1]
    for i in range(0, len(verts) - 1): 
        sum += verts[i][0] * verts[i + 1][1] - verts[i + 1][0] * verts[i][1]
    
    return (1 / 2) * abs(sum)


def pos_angle2vertices(x, y, phi, l, w):
    """
    Params
    ------
    - `x`: x-coordinate 
    - `y`: y-coordinate
    - `phi`: angular orientation 
    - `l`: length of rectangle 
    - `w`: width of rectangle

    Returns
    ------
    - A list of two-element tuples representing the coordinates of the four 
      vertices of this rectangle.
    """
    pos = np.array([x, y])
    forward_vec = (l / 2) * np.array([np.cos(phi), np.sin(phi)])
    side_vec = (w / 2) * np.array([- np.sin(phi), np.cos(phi)])
    front_left = pos + forward_vec + side_vec
    front_right = pos + forward_vec - side_vec
    back_right = pos - forward_vec + side_vec
    back_left = pos - forward_vec - side_vec

    return [front_left, front_right, back_left, back_right]


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

