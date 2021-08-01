from yaml import load, FullLoader
import os
import bpy
from tqdm import tqdm
import logging
import datetime
import random
import mathutils
import numpy as np
from math import radians


def generate_point():
    x = random.uniform(coords['lower_x'], coords['upper_x'])
    y = random.uniform(coords['lower_y'], coords['upper_y'])
    z = random.uniform(coords['lower_z'], coords['upper_z'])
    return mathutils.Vector((x, y, z))


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


def take_snapshot(iteration):
    logger.info("      Centering camera on object")
    # get random points
    point_a = generate_point()
    point_b = generate_point()
    # select camera
    camera = bpy.data.objects['Camera']
    # first point is camera location, difference between both is forward vector
    camera.location = point_a
    forward_vector = point_a - point_b
    # rotate camera along forward vector
    rot_quat = forward_vector.to_track_quat('Z', 'Y')
    camera.rotation_euler = (rot_quat.to_matrix().to_4x4() @ mathutils.Matrix.Rotation(radians(90.0), 4, 'Z')).to_euler()
    # render and save view
    logger.info("      Rendering view")
    bpy.context.scene.render.filepath = os.path.join(path_out, str(iteration))
    bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    print("Initialising Blender processor...")

    # load script config
    with open("C:\\Users\\migue\\PycharmProjects\\mars_dl\\environmentgenerator\\config.yaml") as f:
        config = load(f, Loader=FullLoader)
    path_out = config['blendersnapshots']['path_out'] + datetime.datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")
    coords = config['blendersnapshots']['coordinates']
    no_snaps = config['blendersnapshots']['no_snaps']

    # create output path if it doesn't exist
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # create logger
    logger = logging.getLogger('blender_processor')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(path_out + '/blender_snapshots.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # load scene
    scene = bpy.data.scenes['Scene']
    # make scene the active one
    bpy.context.window.scene = scene
    # delete unecessary objects
    print("Commencing data extraction...")
    # launch snapshot processing
    errors = 0
    for i in tqdm(range(no_snaps), desc='Processing snapshots:'):
        try:
            take_snapshot(i)
        except Exception:
            errors = errors + 1
            logger.error("Fatal error in OBJ processing", exc_info=True)
    if errors == 0:
        print("All input OBJs have been successfully processed. See the logs for more details.")
        logger.info("Execution finished without errors.")
    else:
        print(
            f"All input OBJs have been processed, but there were {str(errors)} errors during the execution of the script, please see the logs for more information.")
        logger.info("Execution finished with {str(errors)} errors.")
