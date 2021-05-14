import os
from glob import glob

from joblib import Parallel, delayed
from tqdm import tqdm
from yaml import load, FullLoader
import numpy as np
from PIL import Image
from pathlib import Path
from math import pi, tan
from scipy.ndimage.filters import gaussian_filter, median_filter
#from scipy.spatial import Delaunay
import sys

# focal length from camera intrinsic matrix (obtained from Blender)
f_x = 2666.665
f_y = 2666.665


# undo distance to depth conversion to avoid fish-eye effect
def undo_conversion(depth_map, camera_fov=50.7):
    img_width = depth_map.shape[1]
    img_height = depth_map.shape[0]
    focal_in_pixels = (img_width * 0.5) / tan(camera_fov * 0.5 * pi / 180)

    # Get x_i and y_i (distances from optical center)
    cx = img_width // 2
    cy = img_height // 2

    xs = np.arange(img_width) - cx
    ys = np.arange(img_height) - cy
    xis, yis = np.meshgrid(xs, ys)

    dist_map = depth_map / focal_in_pixels * np.sqrt(
        xis ** 2 + yis ** 2 + focal_in_pixels ** 2)

    return dist_map


def to_obj(tile, outfile):
    # todo: is this sigma okay?

    # TODO: z has values in [-1,1]. What do the negative values represent?


    # fetch rgb and depth values and undo normalisation
    rgb = (tile[...,:3] / 2) + 0.5
    z = (tile[:, :, 3]+1)*50

    # apply smoothing filters
    z = median_filter(z, size=10)
    z = gaussian_filter(z, sigma=0.5)
    # Todo:interpolation necessary?
    #z = np.interp(z, (z.min(), z.max()), (+0.5, +1))
    #

    ### OUTLIER FILTERING
    # Calculate outlier filtering parameters
    mu = np.mean(z)
    # If height values are mostly negative, invert sign
    if (mu < 0):
        z = -z
        mu = -mu
    sigma = np.std(z)
    ######## TODO: uncomment this (or not - this sets all values to one!)

    ##### TODO IDEA: FILTER OUT DEPTHS GREATER THAN 40
    ###### TODO ALSO: DEPTH DATA MIGHT BE INVERTED
    inliers_idx = np.abs(z - mu) < 2*sigma
    #inliers_idx = np.ones(z.shape, np.bool)
    TOTAL_POINTS = (tile.shape[0] * tile.shape[1])
    num_outliers = TOTAL_POINTS - inliers_idx.sum()
    MAX_OUTLIERS = 0.05 * TOTAL_POINTS
    # If there are many outliers, but within a reasonable treshold, keep tile and don't filter outliers
    #if MAX_OUTLIERS <= num_outliers <= 2 * MAX_OUTLIERS:
    #    print(
    #        f'Skipping outliers filtering because there are {num_outliers} outliers (mu={mu}, std={sigma}, max outliers={MAX_OUTLIERS}).')
    # If number of outliers > 2*max_outliers: discard tile
    # elif num_outliers > 2*MAX_OUTLIERS:
    #     print(f'Tile (mu={mu}, std={sigma}) contains {num_outliers} outliers, which is more than two times the treshold of max outliers({MAX_OUTLIERS}). Discarding tile.')
    #     return

    if is_image:
        z = undo_conversion(z)

    # Flatten z and project depth values to their 3D coordinates
    # two subarrays for x and y coord of length 1024x1024
    idx = np.mgrid[:z.shape[0], :z.shape[1]].reshape([2, -1])
    z = z.reshape([1, -1])

    # todo: parametrise f_x and f_y
    # Directly calculate point coordinates without passing through matrix multiplication (this is more accurate)
    C_alt = np.ones([4, idx.shape[1]])
    C_alt[0, :] = z[0, :] * 1/f_x * idx[0, :]
    C_alt[1, :] = z[0, :] * 1/f_y * idx[1, :]
    C_alt[2, :] = z[0, :]

    # Reshape back to 2D image
    C = C_alt.reshape([4, *tile.shape[:2]])

    # Todo: are really all cases so extreme?
    # Build vertex index-coordinates map for later triangulation
    idx_map = {}
    next_idx = 1
    with outfile.open("w") as obj_file:
        obj_file.write("# Wavefront object file generated by EnvGen\n"
                       f"\nmtllib {outfile.with_suffix('.mtl').name}\n")
        # Add geometric vertices
        point_cloud = []
        for r in range(tile.shape[0]):
            for c in range(tile.shape[1]):
                xyz = C[:3, r, c]
                # Add and write vertex if it's an inlier or the number of outliers isn't above treshold
                if inliers_idx[r,c]: #num_outliers >= MAX_OUTLIERS or
                    idx_map[(r,c)] = next_idx
                    next_idx += 1
                    point_cloud.append(xyz)
                    obj_file.write(f"v {' '.join([np.format_float_positional(el, precision=19, trim='0') for el in xyz])}\n")

        # Write texture vertices
        obj_file.write("\nusemtl material_0\n")
        texture_coordinates = np.asarray(point_cloud)[...,:2] / np.asarray([tile.shape[:2]])
        for uv in texture_coordinates:
            obj_file.write(f"vt {' '.join([np.format_float_positional(el, precision=19, trim='0') for el in uv])}\n")

        # Triangulate and write faces
        obj_file.write("\n")
        for r in range(tile.shape[0]-1):
            for c in range(tile.shape[1]-1):
                f1_idxes = [
                    idx_map.get((r,     c)),
                    idx_map.get((r,   c+1)),
                    idx_map.get((r+1,   c)),
                ]
                f2_idxes = [
                    idx_map.get((r,   c+1)),
                    idx_map.get((r+1, c+1)),
                    idx_map.get((r+1,   c)),
                ]
                if all(f1_idxes):
                    #print('Found f1 triangle')
                    # write f1 triangle
                    obj_file.write(f"f {' '.join(f'{v}/{v}' for v in f1_idxes)}\n")
                if all(f2_idxes):
                    # write f2 triangle
                    #print('Found f2 triangle')
                    obj_file.write(f"f {' '.join(f'{v}/{v}' for v in f2_idxes)}\n")

        # # Delaunay triangulation creates the faces of the object
        # mesh = Delaunay(point_cloud)
        # # Wavefront OBJ files count from index 1, not 0
        # #mesh.simplices += 1
        # for face in mesh.simplices:
        #     line_to_write = "f "
        #     for point in face:
        #         line_to_write += " " + point.astype(str)
        #     file.write(line_to_write+"\n")

        # Write out texture image
        Image.fromarray(np.rint(rgb*255).astype(np.uint8)).convert('RGB').save(outfile.with_suffix('.png'))

    with open(outfile.with_suffix('.mtl'), "w") as mtl_file:
        mtl_file.write("# Wavefront material file generated by EnvGen\n"
                       "newmtl material_0\n"
                       "Ka 0.200000 0.200000 0.200000\n"
                       "Kd 1.000000 1.000000 1.000000\n"
                       "Ks 1.000000 1.000000 1.000000\n"
                       "Tr 1.000000\n"
                       "illum 2\n"
                       "Ns 0.000000\n"
                       f"map_Kd {outfile.with_suffix('.png').name}")



def image_chunker(image):
    # max number of chunks per axis, last chunk on each axis will overlap slightly
    chunks_x = int(np.floor(image.shape[0] / psgan_output_size))
    chunks_y = int(np.floor(image.shape[1] / psgan_output_size))
    chunks = []
    # get indices of chunk, then slice chunk and append
    for i in range(chunks_x):
        for j in range(chunks_y):
            idx_x = [(psgan_output_size * i) + padding_psgan, (psgan_output_size * (i + 1)) + padding_psgan]
            idx_y = [(psgan_output_size * j) + padding_psgan, (psgan_output_size * (j + 1)) + padding_psgan]
            chunks.append(image[idx_x[0]:idx_x[1], idx_y[0]:idx_y[1], :])
    return chunks


def process_mosaic(mosaic_path):
    if is_image:
        # Todo: adapt to relative path
        mosaic = np.asarray(Image.open(mosaic_path))
        a = Image.open(mosaic_path)
        tiles = image_chunker(mosaic)
    else:
        # todo: more elegant solution? is the second move axis necessary?
        tiles = np.moveaxis(np.moveaxis(np.load(path_in / mosaic_path), 1, -1), 1, 2)
    # if object is a mosaic of several tiles
    if tiles.ndim == 4:
        try:
            for i, tile in enumerate(tiles):
                # np.save(f'E:/tile_{i}.npy', tile)
                out_name = path_out / mosaic_path.with_name(mosaic_path.stem + f"_tile{i}.obj")
                out_name.parent.mkdir(parents=True, exist_ok=True)
                to_obj(tile, out_name)
        except Exception as e:
            print("\nException while processing following multiple tile: " + str(mosaic_path) + "\n" + repr(e))
    # if object contains a single tile
    elif tiles.ndim == 3:
        try:
            out_name = path_out / mosaic_path.with_suffix(".obj")
            out_name.parent.mkdir(parents=True, exist_ok=True)
            to_obj(tiles, out_name)
        except Exception as e:
            print("\nException while processing following single tile: " + str(mosaic_path) + "\n" + repr(e))
    else:
        sys.exit('Invalid mosaic shape: '+str(tiles.shape))

if __name__ == '__main__':
    print("Initialising OBJ converter...")

    # load script config
    with open("config.yaml") as f:
        config = load(f, Loader=FullLoader)
    config = config['obj_converter']
    path_in = Path(config['path_in'])
    path_out = path_in / "obj/"
    psgan_output_size = config['psgan_output_size']
    padding_psgan = config['padding_psgan']
    is_image = config['is_image']

    # create output path if it doesn't exist
    if not os.path.exists(path_out):
        print("Output folder created: /obj")
        os.makedirs(path_out)

    # process texture mosaics
    #to_process = [file for file in glob(path_in + "**/*.npy")]
    to_process = [file.relative_to(path_in) for file in path_in.glob("**/*.npy")]
    #for mosaic_path in tqdm(to_process, desc="Converting SGAN results into OBJ format"):
    #    process_mosaic(mosaic_path)

    try:
        Parallel(n_jobs=-1, backend="loky")(
            map(delayed(process_mosaic), (mosaic_path for mosaic_path in tqdm(to_process, desc="Converting SGAN results into OBJ format"))))
    except Exception as e:
        print("Exception while concurrently processing SGAN results: " + repr(e))

    print("Dataset successfully processed.")
