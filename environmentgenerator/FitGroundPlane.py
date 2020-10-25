from yaml import load, FullLoader
import pywavefront
from skspatial.objects import Points, Plane
from skspatial.plotting import plot_3d
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # load object config file
    print("Initialising ground plane fitter...")
    with open("config.yaml") as f:
        config = load(f, Loader=FullLoader)
    path_in = config['obj2heightmap']['path_in']
    obj_in = config['obj2heightmap']['obj_in']

    # read OBJ file
    print("Processing OBJ...")
    scene = pywavefront.Wavefront(r'%s%s' % (path_in, obj_in), create_materials=True, collect_faces=True)

    # TODO:CLEAN THIS UP
    print("Fitting plane...")
    vertices = np.asarray(scene.vertices)
    vertices = Points(scene.vertices)
    # discard given proportion of vertices for plane calculation; if it still doesn't fit in memory,
    # reduce number of vertices used by steps of increasing malloc_step size
    malloc_successful = False
    step = config['fitgroundplane']['malloc_discard_instances']
    while not malloc_successful:
        try:
            random_vertices_idx = np.random.choice(vertices.shape[0], size=(int(np.ceil(len(vertices) * (1 - step)))),
                                                   replace=False)
            random_vertices = vertices[random_vertices_idx, :]
            plane = Plane.best_fit(random_vertices)
        except MemoryError:
            step = step + config['fitgroundplane']['malloc_step']
        else:
            malloc_successful = True

    print("Plane successfully fit using %i%% of instances: %s" % (100 * (1 - step)) % plane)

    plot_3d(
        random_vertices.plotter(c='k', s=50, depthshade=False),
        plane.plotter(alpha=0.2, lims_x=(-5, 5), lims_y=(-5, 5)),
    )

    plt.show()
