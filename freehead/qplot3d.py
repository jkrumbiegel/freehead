import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import freehead as fh


def qplot3d(data, flipyz=True, xinvert=True):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot = ax.plot(*fh.tup3d(data, flipyz=flipyz))

    if xinvert:
        ax.invert_xaxis()

    return fig, ax, plot