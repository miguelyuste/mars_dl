# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 08:39:50 2020

@author: miguel.fernandez
"""

from matplotlib import pyplot as plt
from wavefront_reader import read_wavefront
import numpy as np
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D

obj_in = "2023.obj"
file_out = "2023.png"
# heightmap_out = "marstest.r16"
path_in = "D:/Master Thesis/craters/source/2023/"

# read OBJ file, extract vertices
print("Processing OBJ...")
geoms = read_wavefront(r'%s%s' % (path_in, obj_in))
cube_vertices = geoms[list(geoms.keys())[0]]['v']
# ranges = np.ceil(np.amax(cube_vertices, 0) - np.amin(cube_vertices, 0)).astype(str)

print("Initating OBJ to heightmap conversion...")

# subprocess.call([path+"obj2hmap", path+obj_in, path+heightmap_out, ranges[0], ranges[1], ranges[2], "y", "tf32"])
# =============================================================================
# os.system("powershell.exe [./%s/obj2hmap mars.obj %s.r16 %i %i %i y tf32]" % (path, filename, ranges[0], ranges[1], ranges[2]))
# =============================================================================

# swap Y and Z axis
# ToDo: assess if inverting axis is needed for all files
#cube_vertices[:, [1, 2]] = cube_vertices[:, [2, 1]]

# extract individual data points
x = cube_vertices[:, 0]
y = cube_vertices[:, 1]
z = cube_vertices[:, 2]

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
plt.savefig('%s%s' % (path_in, file_out))
