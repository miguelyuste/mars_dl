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
from scipy.spatial import Delaunay

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


    rgb = (tile[...,:3] / 2) + 0.5 # tile has values in [-1,1], map RGB to [0,1]
    z = tile[:, :, 3]
    z = median_filter(z, size=9)

    if is_image:
        z = undo_conversion(z)

    # Flatten z and project depth values to their 3D coordinates

    idx = np.mgrid[:z.shape[0], :z.shape[1]].reshape([2, -1])
    z = z.reshape([1, -1])
    V = np.vstack([idx, np.ones_like(z), 1 / z])

    # todo: parametrise f_x and f_y
    M = np.eye(4)
    M[0, 0] = 0.000375
    M[1, 1] = 0.000375

    # 3D coordinates of points (flattened)
    C = (z * (M @ V)).reshape([-1, z.shape[0], z.shape[1]])

    # Undo flattening (TODO: Check that the reshaping is in the correct order)
    C = C.squeeze().reshape([4, *tile.shape[:2]])

    point_cloud = []
    idx_map = {}
    next_idx = 1
    with open(path_out + outfile, "w") as file:
        # Extract points and RGB values
        for r in range(tile.shape[0]):
            for c in range(tile.shape[1]):
                xyz = C[:3, r, c]
                if 0 < xyz[2] < 40:
                    idx_map[(r,c)] = next_idx
                    next_idx += 1
                    point_cloud.append(xyz)
                    rgb_val = rgb[r,c,:].astype(str)
                    xyz = xyz.astype(str)
                    file.write(
                        "v " + xyz[0] + " " + xyz[1] + " " + xyz[2] + " " + rgb_val[0] + " " + rgb_val[1] + " " + rgb_val[2] + "\n")
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
                    file.write(f"f {' '.join(str(v) for v in f1_idxes)}\n")
                if all(f2_idxes):
                    # write f2 triangle
                    #print('Found f2 triangle')
                    file.write(f"f {' '.join(str(v) for v in f2_idxes)}\n")
        Image.fromarray(np.rint(rgb*255).astype(np.uint8)).convert('RGB').save(Path(path_out + outfile).with_suffix('.png'))


        # # Delaunay triangulation creates the faces of the object
        # mesh = Delaunay(point_cloud)
        # # Wavefront OBJ files count from index 1, not 0
        # #mesh.simplices += 1
        # for face in mesh.simplices:
        #     line_to_write = "f "
        #     for point in face:
        #         line_to_write += " " + point.astype(str)
        #     file.write(line_to_write+"\n")



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
        mosaic = np.asarray(Image.open(mosaic_path))
        a = Image.open(mosaic_path)
        tiles = image_chunker(mosaic)
    else:
        # todo: more elegant solution? is the second move axis necessary?
        tiles = np.moveaxis(np.moveaxis(np.load(mosaic_path), 1, -1), 1, 2)
    for i, tile in enumerate(tiles):
        # np.save(f'E:/tile_{i}.npy', tile)
        to_obj(tile, Path(mosaic_path).stem + f"_{i}.obj")


if __name__ == '__main__':
    print("Initialising OBJ converter...")

    # load script config
    with open("config.yaml") as f:
        config = load(f, Loader=FullLoader)
    config = config['obj_converter']
    path_in = config['path_in']
    path_out = path_in + "/obj/"
    psgan_output_size = config['psgan_output_size']
    padding_psgan = config['padding_psgan']
    is_image = config['is_image']

    # create output path if it doesn't exist
    if not os.path.exists(path_out):
        print("Output folder created: /obj")
        os.makedirs(path_out)

    # process texture mosaics
    to_process = [file for file in glob(path_in + "**/generated_textures_089*.npy")]
    for mosaic_path in tqdm(to_process, desc="Converting PSGAN results into OBJ format"):
        process_mosaic(mosaic_path)

    print("Dataset successfully processed.")
