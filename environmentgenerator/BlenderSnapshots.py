from yaml import load, FullLoader
import os
import bpy
from tqdm import tqdm
import logging
import datetime
import random

def generate_point():
    x = random.uniform(coords['lower_x'], coords['upper_x'])
    y = random.uniform(coords['lower_y'], coords['upper_y'])
    z = random.uniform(coords['lower_z'], coords['upper_z'])
    return [x,y,z]

def process_obj(object_key):
    # if object isn't a light source, terrain nor camera
    logger.info("Processing OBJ: " + object_key)
    if "light" not in object_key.lower() and "generated" not in object_key.lower() and "camera" not in object_key.lower():
        # select object
        logger.info("      Selecting object")
        obj = bpy.data.objects[object_key]
        # center camera on object
        logger.info("      Centering camera on object")
        bpy.ops.view3d.camera_to_view_selected()
        logger.info("      Rendering object")
        bpy.context.scene.render.filepath = os.path.join(path_out, object_key)
        bpy.ops.render.render(write_still=False)
    else:
        logger.info("      OBJ isn't a shatter cone, skipping it")


if __name__ == '__main__':
    print("Initialising Blender processor...")

    # load script config
    with open("C:\\Users\\migue\\PycharmProjects\\mars_dl\\environmentgenerator\\config.yaml") as f:
        config = load(f, Loader=FullLoader)
    path_out = config['blendersnapshots']['path_out'] + datetime.datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")
    coords = config['blendersnapshots']['coordinates']

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

    # # add lighting
    # logger.info("Adding lighting")
    # # create light datablock, set attributes
    # light_data = bpy.data.lights.new(name="light_2.80", type='POINT')
    # light_data.energy = 100

    # create new object with our light datablock
    # light_object = bpy.data.objects.new(name="light_2.80", object_data=light_data)

    # link light object
    # bpy.context.collection.objects.link(light_object)

    # make it active
    # bpy.context.view_layer.objects.active = light_object

    # launch OBJ processing
    errors = 0
    for object_key in tqdm(bpy.data.objects.keys(), desc='Processing shatter cones:'):
        try:
            process_obj(object_key)
        except Exception:
            error = error + 1
            logger.error("Fatal error in OBJ processing", exc_info=True)
    if errors == 0:
        print("All input OBJs have been successfully processed. See the logs for more details.")
        logger.info("Execution finished without errors.")
    else:
        print(
            f"All input OBJs have been processed, but there were {str(errors)} errors during the execution of the script, please see the logs for more information.")
        logger.info("Execution finished with {str(errors)} errors.")
