# -*- coding: utf-8 -*-

import os.path
import json
import numpy as np
import argparse
from copy import deepcopy

# params with default values
num_snaps: int = 1000
radius = 250
up = r'"up": "[1.000,0.000,0.000]"'

# JSON outfile header
out_dict = {
    "fieldOfView": 5.47,
    "resolution": "[1024, 1024]",
    "snapshots": [],
    "version": 0
}

# structure of a single snapshot
template_snapshot = {
    "filename": "",
    "view": {
        "forward": [],
        "location": [],
        "up": []
    },
    "surfaceUpdates": []
}

# structure of an object update
template_update = {
    "opcname": "",
    "visible": False
}

def sample_spherical(npoints, radius, obj, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    vec *= radius
    locations = vec + np.expand_dims(obj, -1)
    locations[2, :] = 162.46
    forwards = np.expand_dims(obj, -1) - locations
    return locations.T, forwards.T

def create_snapshots():
    snaps = []
    # generate the given number of snapshots for each camera center
    for center in objects['cameraCenters']:
        # for each randomly generated point of the circumference
        loc, fwd = sample_spherical(num_snaps, radius, center['center'])
        for i, (forward, up) in enumerate(zip(fwd, loc)):
            # 1) generate snapshot without shatter cones but with scenes
            sample = deepcopy(template_snapshot)
            sample['filename'] = str(i) + "_OPCs"
            sample['view']['forward'] = forward.tolist()
            sample['view']['location'] = center['center']
            sample['view']['up'] = up.tolist()
            for cone in objects['shatterCones']:
                cone_update = deepcopy(template_update)
                cone_update['opcname'] = cone['coneName']
                # add trafo if given
                if "coneTrafo" in cone:
                    cone_update['trafo'] = cone['coneTrafo']
                sample['surfaceUpdates'].append(cone_update)
            for opc in objects['opcs']:
                opc_update = deepcopy(template_update)
                opc_update['opcname'] = opc['opcName']
                opc_update['visible'] = True
                sample['surfaceUpdates'].append(opc_update)
            snaps.append(deepcopy(sample))
            # 2) generate snapshot with shatter cones but without scenes
            sample = deepcopy(template_snapshot)
            sample['filename'] = str(i) + "_SCs"
            sample['view']['forward'] = forward.tolist()
            sample['view']['location'] = center['center']
            sample['view']['up'] = up.tolist()
            for cone in objects['shatterCones']:
                cone_update = deepcopy(template_update)
                cone_update['opcname'] = cone['coneName']
                cone_update['visible'] = True
                sample['surfaceUpdates'].append(cone_update)
            for opc in objects['opcs']:
                opc_update = deepcopy(template_update)
                opc_update['opcname'] = opc['opcName']
                sample['surfaceUpdates'].append(opc_update)
            snaps.append(deepcopy(sample))
    return snaps

if __name__ == '__main__':
    # optional argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="output folder where to place the generated configuration; location of script by default")
    parser.add_argument("-f", "--filename", help="name of generated file; snapshot_config by default")
    parser.add_argument("-r", "--radius", type=float, help="radius of camera circular trajectory; 250 by default")
    parser.add_argument("-s", "--snaps", type=int, help="number of snapshots to generate for each center; 1000 by default")
    parser.add_argument("-res", "--resolution", help="number of snapshots to generate for each center; [1024,1024] by default")
    parser.add_argument("-fov", "--fieldofview", type=float, help="field of view; 5.47 by default")
    args = parser.parse_args()
    if args.output:
        out_path = args.output
    else: out_path = "../"
    if args.filename:
        out_path += args.filename
    else: out_path += "/snapshots_config"
    if args.radius:
        radius = args.radius
    if args.snaps:
        num_snaps = args.snaps
    if args.resolution:
        out_dict['resolution'] = args.resolution
    if args.fieldofview:
        out_dict['fieldOfView'] = args.fieldofview

    # load object config file
    with open("ObjectConfiguration.json") as objects_json:
        objects = json.load(objects_json)

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
