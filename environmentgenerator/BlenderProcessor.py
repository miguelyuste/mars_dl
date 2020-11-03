from yaml import load, FullLoader
# import pywavefront
from skspatial.objects import Points, Plane
from skspatial.plotting import plot_3d
import numpy as np
import matplotlib.pyplot as plt
import os
import bpy
import mathutils


def delete_scene_objects():
    # todo: delete light and camera??
    #    scene = bpy.context.window.scene
    #    for object_ in scene.objects:
    #        if(object_.name!="Camera"):
    #            bpy.data.objects.remove(object_, True)

    # todo: object and scene cleanup do not work! camera isnt deleted though, that works
    for o in bpy.context.scene.objects:
        if o.type != 'CAMERA':
            o.select_set(True)
        else:
            o.select_set(False)
    # bpy.data.objects['Camera'].select_set(False)
    bpy.ops.object.delete(True)

    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    bpy.ops.wm.open_mainfile(filepath=bpy.data.filepath)


def fit_plane(vertices):
    # TODO:CLEAN THIS UP
    print("Fitting plane...")
    #    vertices = np.asarray(scene.vertices)
    vertices = Points(vertices)
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

    print(f"Plane successfully fit using {100 * (1 - step)}% of instances: {plane}")

    #    plot_3d(
    #        random_vertices.plotter(c='k', s=50, depthshade=False),
    #        plane.plotter(alpha=0.2, lims_x=(-5, 5), lims_y=(-5, 5)),
    #    )

    #    plt.show()

    return plane

# todo: replace hardcoded values, rename class

def reverse_mapping(array, from_min=0, from_max=1, to_min=0.1, to_max=100):
    slope = (to_max - to_min) / (from_max - from_min)
    return to_min + slope * (array - from_min)


def get_depthmap():  # ACTUALLY, this is the DISTANCE map (distance of each pixel from the camera!)
    scene = bpy.data.scenes['Scene']
    tree = scene.node_tree
    # create output node
    v = tree.nodes.new('CompositorNodeViewer')
    v.use_alpha = False
    depth = np.asarray(bpy.data.images["Viewer Node"].pixels)
    depth = np.reshape(depth, (1080, 1920, 4))
    depth = depth[:, :, 0]
    depth = np.flipud(depth)
    depth = reverse_mapping(depth)

    # np.save(r"C:\Users\migue\Desktop/distance_from_camera.npy", depth)
    return depth


def distance_to_depth_conversion(dist_img, camera_fov=50.7):
    img_width = dist_img.shape[1]
    img_height = dist_img.shape[0]
    focal_in_pixels = (img_width * 0.5) / tan(camera_fov * 0.5 * pi / 180)

    # Get x_i and y_i (distances from optical center)
    cx = img_width // 2
    cy = img_height // 2

    xs = np.arange(img_width) - cx
    ys = np.arange(img_height) - cy
    xis, yis = np.meshgrid(xs, ys)

    depth = np.sqrt(
        dist_img ** 2 / (
                (xis ** 2 + yis ** 2) / (focal_in_pixels ** 2) + 1
        )
    )
    return depth

def process_obj(filepath):
    delete_scene_objects()
    keys_before_import = bpy.data.objects.keys()
    bpy.ops.import_scene.obj(filepath=filepath)
    imported_keys = [s for s in bpy.data.objects.keys() if s not in keys_before_import]
    print(keys_before_import)
    # object = bpy.context.selected_objects[0] ####<--Fix
    object = bpy.data.objects[imported_keys[0]]  ####<--Fix
    object.rotation_euler = mathutils.Vector([0, 0, 0])  # clear any possible rotation set in the obj
    print('Imported name: ', object.name)
    # print(object.data.vertices.values())
    vertices = np.empty([len(bpy.data.meshes[object.name].vertices), 3], dtype=np.float32)
    for i, vertex in enumerate(bpy.data.meshes[object.name].vertices):
        vertices[i] = np.asarray(vertex.co)
    estimated_center_point = vertices.mean(axis=0)

    # print(np.asarray(bpy.data.meshes[object.name].vertices[0].co))
    # ToDo: is point centered? otherwise, average all vertices
    plane = fit_plane(vertices)
    camera = bpy.data.objects['Camera']
    # todo: greater camera distance
    camera.location = estimated_center_point + 5 * np.asarray(plane.normal)
    looking_direction = camera.location - mathutils.Vector(estimated_center_point)
    rot_quat = looking_direction.to_track_quat('Z', 'Y')

    camera.rotation_euler = rot_quat.to_euler()

    get_depthmap()


if __name__ == '__main__':
    file_loc = 'E:\\TFM\\Bakeoff 2020\\Job_0796_003477_MSLMST_obj_cart_single\\T000_P007_C000.obj'
    # load object config file
    print("Initialising ground plane fitter...")
    with open("C:\\Users\\migue\\PycharmProjects\\mars_dl\\environmentgenerator\\config.yaml") as f:
        config = load(f, Loader=FullLoader)
    path_in = config['obj2heightmap']['path_in']
    obj_in = config['obj2heightmap']['obj_in']

    # load scene
    scene = bpy.data.scenes['Scene']
    # make scene the active one
    bpy.context.window.scene = scene
    process_obj(file_loc)


