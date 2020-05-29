# mars_dl
Collection of scripts created for Joanneum Research's Mars Deep Learning project.

## Random Pose Generator
Generates random configurations for PRo3D automatic snapshots around a series of camera centres. Two configurations are exported for each random snapshot point:
1. Snapshot with hidden shatter cones and visible scenery
2. Snapshot with visible shatter cones and hidden scenery
#### Usage 
The following parameters must be specified in `ObjectConfiguration.json`:
- The names of the (scenery) OPCs 
- The names, locations, and optionally transformations of the shatter cones
- The chosen camera centers 

The script can then be run through a command line and optional parameters:

`RandomPoseGenerator.py [-h] [-o OUTPUT] [-f FILENAME] [-r RADIUS] [-s SNAPS] [-res RESOLUTION] [-fov FIELDOFVIEW]` 
