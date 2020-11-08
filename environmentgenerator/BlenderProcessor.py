import os
from math import pi, tan
from pathlib import Path

import bpy
import mathutils
import numpy as np
from sklearn.decomposition import PCA
from skspatial.objects import Points, Plane
from tqdm import tqdm
from yaml import load, FullLoader


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


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


def reverse_mapping(array, from_min=0, from_max=1, to_min=0.1, to_max=100):
    slope = (to_max - to_min) / (from_max - from_min)
    return to_min + slope * (array - from_min)


# distance map from camera to surfaces
def get_distancemap_rgbimage():
    scene = bpy.data.scenes['Scene']
    tree = scene.node_tree
    # create output node
    # v = tree.nodes.new('CompositorNodeViewer')
    # v.use_alpha = False

    tree.nodes['Switch'].check = True
    bpy.ops.render.render(write_still=False)
    distance_map = np.asarray(bpy.data.images["Viewer Node"].pixels)

    # todo: if camera resolution can be tweaked, this can be parametrised
    distance_map = np.reshape(distance_map, (1080, 1920, 4))
    distance_map = distance_map[:, :, 0]
    distance_map = np.flipud(distance_map)
    distance_map = reverse_mapping(distance_map)

    tree.nodes['Switch'].check = False
    bpy.ops.render.render(write_still=False)
    rgb_image = np.asarray(bpy.data.images["Viewer Node"].pixels)

    # todo: if camera resolution can be tweaked, this can be parametrised
    rgb_image = np.reshape(rgb_image, (1080, 1920, 4))
    rgb_image = rgb_image[:, :, 0:3]
    rgb_image = np.flipud(rgb_image)
    rgb_image = reverse_mapping(rgb_image)

    return distance_map, rgb_image


def distance_to_depth_conversion(dist_map, camera_fov=50.7):
    img_width = dist_map.shape[1]
    img_height = dist_map.shape[0]
    focal_in_pixels = (img_width * 0.5) / tan(camera_fov * 0.5 * pi / 180)

    # Get x_i and y_i (distances from optical center)
    cx = img_width // 2
    cy = img_height // 2

    xs = np.arange(img_width) - cx
    ys = np.arange(img_height) - cy
    xis, yis = np.meshgrid(xs, ys)

    depth_map = np.sqrt(
        dist_map ** 2 / (
                (xis ** 2 + yis ** 2) / (focal_in_pixels ** 2) + 1
        )
    )

    return depth_map


def process_obj(filepath):
    # import object
    keys_before_import = bpy.data.objects.keys()
    bpy.ops.import_scene.obj(filepath=filepath)
    # todo: is this necessary?
    imported_keys = [s for s in bpy.data.objects.keys() if s not in keys_before_import]
    # select object
    object = bpy.data.objects[imported_keys[0]]
    # clear any possible rotation set in the obj
    object.rotation_euler = mathutils.Vector([0, 0, 0])
    print('Processing obj: ', object.name)
    # extract vertices
    vertices = np.empty([len(bpy.data.meshes[object.name].vertices), 3], dtype=np.float32)
    for i, vertex in enumerate(bpy.data.meshes[object.name].vertices):
        vertices[i] = np.asarray(vertex.co)
    # calculate center point and PCA for camera placing
    estimated_center_point = vertices.mean(axis=0)
    pca = PCA(n_components=3)
    pca.fit(vertices)
    V = pca.components_
    width_comp = V.T[0]
    normal_comp = V.T[2]
    camera = bpy.data.objects['Camera']
    # point camera at object central point
    camera.location = estimated_center_point + np.asarray(normal_comp)
    # rotate and align camera with object
    looking_direction = camera.location - mathutils.Vector(estimated_center_point)
    rot_quat = looking_direction.to_track_quat('Z', 'Y')
    cam_x = rot_quat @ mathutils.Vector((1.0, 0.0, 0.0))
    roll = angle_between(width_comp, np.asarray(cam_x))
    camera_roll = mathutils.Matrix.Rotation(roll, 4, 'Z').to_quaternion()
    camera.rotation_euler = (rot_quat @ camera_roll).to_euler()
    # close up on object (fit entire object into view)
    bpy.ops.view3d.camera_to_view_selected()
    # get distance to camera and RGB values
    distance_map, rgb_image = get_distancemap_rgbimage()
    # delete object so it isn't visible in the next image
    bpy.ops.object.select_all(action='DESELECT')
    object.select_set(True)
    bpy.ops.object.delete()
    # convert camera distances to actual depth data
    depth_map = distance_to_depth_conversion(distance_map)
    # prepare and save as numpy array
    rgbd_image = np.concatenate([rgb_image, depth_map[..., None]], axis=-1)
    np.save(str(path_out + Path(filepath).stem + ".npy"), rgbd_image)


if __name__ == '__main__':
    file_loc = 'E:\\TFM\\Bakeoff 2020\\Job_0796_003477_MSLMST_obj_cart_single\\T000_P007_C000.obj'
    # load object config file
    print("Initialising Blender processor...")
    with open("C:\\Users\\migue\\PycharmProjects\\mars_dl\\environmentgenerator\\config.yaml") as f:
        config = load(f, Loader=FullLoader)
    path_in = config['blenderprocessor']['path_in']
    path_out = config['blenderprocessor']['path_out']
    # create output path if it doesn't exist
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    # load scene
    scene = bpy.data.scenes['Scene']
    # make scene the active one
    bpy.context.window.scene = scene
    # delete unecessary objects
    delete_scene_objects()
    print("Commencing data extraction...")
    for file in tqdm(Path(path_in).rglob('*.obj'), desc='Processing OBJs:'):
        process_obj(str(file))
