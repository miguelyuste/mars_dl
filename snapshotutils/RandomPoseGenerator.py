# -*- coding: utf-8 -*-

import os.path
import json
import numpy as np
import argparse
from copy import deepcopy
import yaml

# JSON outfile header with default PRo3D params
out_dict = {
    "fieldOfView": 5.47,
    "resolution": "[1024, 1024]",
    "snapshots": [],
    "version": 0
}

up_vector = np.array([-0.733, 0.675, -0.082])

# structure of a single snapshot
template_snapshot = {
    "filename": "",
    "view": {
        "forward": [],
        "location": [],
        "up": "[-0.733,0.675,-0.082]"
    },
    #"surfaceUpdates": [],
    "shattercones": []
}

# structure of an object update
template_update = {
    "opcname": "",
    "visible": False
}


def sample_spherical(npoints, radius, obj, ndim=3):
    #generate points and normalise them
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    #apply radius
    vec *= radius
    #center points around reshaped interest point
    locations = vec + np.expand_dims(obj, -1)
    #calculate forward vectors and normalise them
    forwards = np.expand_dims(obj, -1) - locations
    forwards_norms = np.linalg.norm(forwards, axis=0)
    forwards /= forwards_norms
    # make forward vectors orthogonal to up
    forwards *= -(np.dot(up_vector, forwards) / forwards_norms * 2)
    forwards = forwards + up_vector[:, None]
    return locations.T, forwards.T


def create_snapshots():
    snaps = []
    # generate the given number of snapshots for each camera center
    for i, center in enumerate(objects['camera_centers']):
        camera_prefix = "cam" + str(i) + "_"
        # for each randomly generated point of the circumference
        loc, fwd = sample_spherical(num_snaps, radius, center)
        for j, (forward, location) in enumerate(zip(fwd, loc)):
            #####todo: add transformation IFF present in file
            sample = deepcopy(template_snapshot)
            sample['view']['forward'] = str(forward.tolist())
            sample['view']['location'] = str((location + 2 * up_vector).tolist())
            sample['filename'] = camera_prefix + str(j)
            # add shatter cones with random count in interval (min, max)
            for cone in objects['shatter_cones']:
                sample['shattercones'].append(dict(name=str(cone), count=np.random.randint(
                    low=int(objects['shatter_cone_settings']['min']),
                    high=int(objects['shatter_cone_settings']['max']))))
            # Deactivated: toggling surfaces on and off
            # 1) generate snapshot without shatter cones but with scenes
            #sample['filename'] = camera_prefix + str(j) + "_OPCs"
            # for cone in objects['shatter_cones']:
            #     cone_update = deepcopy(template_update)
            #     cone_update['opcname'] = cone
            #     # add trafo if given
            #     # if "coneTrafo" in cone:
            #     #     cone_update['trafo'] = str(cone['coneTrafo'])
            #     # sam
            #     # ple['surfaceUpdates'].append(cone_update)
            # for opc in objects['scenes']:
            #     opc_update = deepcopy(template_update)
            #     opc_update['opcname'] = opc
            #     opc_update['visible'] = True
            #     sample['surfaceUpdates'].append(opc_update)
            # snaps.append(deepcopy(sample))
            # # 2) generate snapshot with shatter cones but without scenes
            # sample = deepcopy(template_snapshot)
            # sample['filename'] = camera_prefix + str(j) + "_SCs"
            # sample['view']['forward'] = str(forward.tolist())
            # ##################################
            # sample['view']['location'] = str((2 * up_vector).tolist())
            # # sample['view']['up'] = str(up.tolist())
            # for cone in objects['shatter_cones']:
            #     cone_update = deepcopy(template_update)
            #     cone_update['opcname'] = cone
            #     cone_update['visible'] = True
            #     sample['surfaceUpdates'].append(cone_update)
            # for opc in objects['scenes']:
            #     opc_update = deepcopy(template_update)
            #     opc_update['opcname'] = opc
            #     sample['surfaceUpdates'].append(opc_update)
            snaps.append(deepcopy(sample))
    return snaps


if __name__ == '__main__':
    # todo: default params
    # optional argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output",
                        help="output folder where to place the generated configuration; location of script by default")
    parser.add_argument("-f", "--filename", help="name of generated file; snapshot_config by default")
    parser.add_argument("-r", "--radius", type=float, help="radius of camera circular trajectory; 25 by default")
    parser.add_argument("-s", "--snaps", type=int,
                        help="number of snapshots to generate for each center; 1000 by default")
    parser.add_argument("-res", "--resolution",
                        help="number of snapshots to generate for each center; [1024,1024] by default")
    parser.add_argument("-fov", "--fieldofview", type=float, help="field of view; 5.47 by default")
    args = parser.parse_args()
    # params take default values for snapshot generation
    if args.output:
        out_path = args.output
    else:
        out_path = "../"
    if args.filename:
        out_path += args.filename
    else:
        out_path += "/snapshots_config"
    if args.radius:
        radius = args.radius
    else:
        radius = 25
    if args.snaps:
        num_snaps = args.snaps
    else:
        num_snaps: int = 1000
    # todo: move setting of rs and fov here
    if args.resolution:
        out_dict['resolution'] = args.resolution
    if args.fieldofview:
        out_dict['fieldOfView'] = args.fieldofview

    # load object config file
    with open('snapshotutils/config.yaml') as f:
        objects = yaml.load(f, Loader=yaml.FullLoader)

    # produce snapshots and store them
    out_dict['snapshots'] = create_snapshots()
    out_string = json.dumps(out_dict)

    # find an unused filename
    if os.path.exists(f"{out_path}.json"):
        i = 2
        while os.path.exists("{}_{}.json".format(out_path, i)):
            i += 1
        out_path += "_" + str(i)
    out_path += ".json"
    # write JSON outfile
    with open(out_path, "w+") as out_file:
        out_file.write(out_string)
    print("JSON configuration successfully generated in target directory: {}".format(out_path))
