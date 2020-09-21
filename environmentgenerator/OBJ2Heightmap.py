# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 08:39:50 2020

@author: miguel.fernandez
"""
from matplotlib import pyplot as plt
from wavefront_reader import read_wavefront
import numpy as np
from yaml import load, FullLoader


def to_r16():
    # ToDo: parametrise Powershell call params
    import subprocess
    r16_out = config['obj2heightmap']['r16_out']
    converter_path = config['obj2heightmap']['path_to_converter']
    ranges = np.ceil(np.amax(cube_vertices, 0) - np.amin(cube_vertices, 0)).astype(str)
    subprocess.call(
        [converter_path + "obj2hmap", path_in + obj_in, converter_path + r16_out, ranges[0], ranges[1], ranges[2], "y",
         "tf32"])


def plot_heightmap():
    print("Initating OBJ to heightmap conversion...")
    # import necessary libraries
    from scipy.spatial import Delaunay
    from mpl_toolkits.mplot3d import Axes3D
    heightmap_out = config['heightmap_out']
    # use Delaunay 2D algorithm to triangulate X,Y
    pts = np.delete(cube_vertices, 2, 1)
    print("Calculating Delaunay triangulation...")
    mesh = Delaunay(pts)
    # plot heightmap of (x,y,z) from bird's-eye view filled with the calculated triangulation
    print("Creating heightmap...")
    fig = plt.figure(dpi=1200)
    ax = fig.add_subplot(1, 1, 1, projection='3d', elev=90, azim=0)
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0, 0)
    ax.grid(False)
    ax.plot_trisurf(x, y, z, triangles=mesh.simplices, cmap=plt.cm.Spectral)
    plt.savefig('%s%s' % (path_in, heightmap_out))


if __name__ == '__main__':
    # load object config file
    print("Loading configuration...")
    with open('./config.yaml') as f:
        config = load(f, Loader=FullLoader)
    config = config['obj2heightmap']
    path_in = config['path_in']
    obj_in = config['file_out']

    # read OBJ file, extract vertices
    print("Processing OBJ...")
    geoms = read_wavefront(r'%s%s' % (path_in, obj_in))
    cube_vertices = geoms[list(geoms.keys())[0]]['v']

    # swap Y and Z axis
    # ToDo: assess if inverting axes is needed for all files
    # cube_vertices[:, [1, 2]] = cube_vertices[:, [2, 1]]

    # extract individual data points
    x = cube_vertices[:, 0]
    y = cube_vertices[:, 1]
    z = cube_vertices[:, 2]
