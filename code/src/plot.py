import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from os.path import join

def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=1, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def plot_swimmer_trajectory(positions, 
                            output_dir, 
                            title="Swimmer Trajectory", 
                            t_start='0', 
                            t_stop='t_stop', 
                            xlim=None, 
                            ylim=None):
    fig, ax = plt.subplots()
    path = mpath.Path(positions)
    verts = path.vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.linspace(0, 1, len(x))
    lc = colorline(x, y, z, cmap=plt.get_cmap('jet'), linewidth=1)
    plt.gca().set_aspect('equal')

    plt.annotate("t=0", (x[0], y[0]),  backgroundcolor='w')
    plt.annotate(f"t={t_stop}", (x[-1], y[-1]), backgroundcolor='w')
    plt.title(title, y=1.05)

    axcb = fig.colorbar(lc)
    axcb.set_label('Line Number')

    if xlim is not None: 
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None: 
        ax.set_ylim(ylim[0], ylim[1])
        
    plt.savefig(output_dir)