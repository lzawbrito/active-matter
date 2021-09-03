from src.swimmer import SimpleSwimmer, Swimmer 
from src.world import World 
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.collections as mcoll
import matplotlib.path as mpath

DT = 0.01
DURATION = 500

world = World(DT)
swimmer = SimpleSwimmer(0, world)

positions = []
for i in range(0, int(DURATION / DT)):
    positions.append(swimmer.get_position())
    swimmer.step()

x, y = np.transpose(positions)

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


fig, ax = plt.subplots()
path = mpath.Path(positions)
verts = path.vertices
x, y = verts[:, 0], verts[:, 1]
z = np.linspace(0, 1, len(x))
lc = colorline(x, y, z, cmap=plt.get_cmap('jet'), linewidth=1)
plt.gca().set_aspect('equal')

axcb = fig.colorbar(lc)
axcb.set_label('Line Number')

plt.savefig('plots/swimmer_trajectory.png')