import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import os
import subprocess


def solve_heat_implicit(
        k,
        init_cond,
        low_bdy_cond,
        high_bdy_cond,
        x_domain: list,
        t_stop,
        incx=0.01,
        inct=0.01,
        th=0.5
):
    """Solves the heat equation using the implicit scheme. Defaults to the
     Crank-Nicolson scheme."""

    try:
        xlow, xhigh = x_domain

        r = (inct * k) / incx ** 2

        tsteps = int(t_stop / inct)
        xsteps = int((xhigh - xlow) / incx)
        print(f"tsteps, xsteps: {tsteps}, {xsteps}")

        x = np.linspace(xlow, xhigh, xsteps)
        t = np.linspace(0, t_stop, tsteps)

        a = []
        for i in range(1, xsteps - 1):
            new_row = []
            for j in range(1, xsteps - 1):
                if i == j:
                    new_row.append(1 + 2 * r * th)
                elif i == j - 1 or i == j + 1:
                    new_row.append(-1 * r * th)
                else:
                    new_row.append(0)
            a.append(new_row)

        a_matrix = np.array(a)
        print(a_matrix)

        def find_b(row):
            b = []
            for i in range(1, xsteps - 1):
                entry = (1 - th) * r * row[i - 1] + (1 - 2 * (1 - th) * r) * row[i] \
                        + (1 - th) * r * row[i + 1]
                b.append(entry)
            return np.array(b)

        def solve_new_row(b, current_t):
            solved = linalg.solve(a_matrix, find_b(b))
            bdys = np.insert(solved, [0, len(solved)],
                             [low_bdy_cond(current_t), high_bdy_cond(current_t)])
            return bdys

        u = [init_cond(x)]
        for i in range(1, len(t)):
            u.append(solve_new_row(u[i - 1], t[i]))

        return np.array(u), np.array(x), np.array(t)
    except ValueError:
        print("Error: x domain should be an iterable of length 2.")


u_, x, t_ = solve_heat_implicit(1., lambda x: x, lambda x: 0, lambda x: 0, [-2, 2], 2.0, 0.01, 0.01)


t = np.resize(t_, x.shape)
print(f"shape: {x.shape}")
u = np.transpose(np.array([np.resize(row, x.shape) for row in np.transpose(u_)]))
print(f"ushape: {u.shape}")


def make_animation(input, function):
    fname = os.path.basename(__file__)

    fig, ax = plt.subplots()

    for i in range(0, len(function)):
        ax.plot(input, function[i])
        ax.set_ylim(min(u[0]) - 0.1, max(u[0]) + 0.1)
        my_path = os.path.abspath(__file__)
        plt.savefig(my_path[:-len(fname)] + '/frames/frame' + str(i + 1).zfill(3) + '.jpg')
        plt.cla()

    cmd = ['ffmpeg', '-i', my_path[:-len(fname)] + '/frames/frame%03d.jpg', 'output.mov']
    retcode = subprocess.call(cmd)
    if not retcode == 0:
        raise ValueError('Error {} executing command: {}'.format(retcode, ' '.join(cmd)))


make_animation(x, u)
