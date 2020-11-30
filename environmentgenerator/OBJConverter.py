import os
from glob import glob

from joblib import Parallel, delayed
from tqdm import tqdm
from yaml import load, FullLoader
import numpy as np
from PIL import Image
from pathlib import Path
from math import pi, tan

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
    #data = np.random.random([600, 800])
    z = tile[:, :, 3]
    z = undo_conversion(z)

    idx = np.mgrid[:z.shape[0], :z.shape[1]].reshape([2, -1])
    z = z.reshape([1, -1])
    V = np.vstack([idx, np.ones_like(z), 1 / z])

    # todo: parametrise f_x and f_y
    M = np.eye(4)
    M[0, 0] = 0.000375
    M[1, 1] = 0.000375

    C = (z * (M @ V)).reshape([-1, z.shape[0], z.shape[1]])

    RGBD = np.empty([z.shape[0], z.shape[1], 4])

    p = RGBD[0, 0, :]
    C[:, 0, 0]

    with open(path_out + outfile, "w") as file:
        for r in range(z.shape[0]):
            for c in range(z.shape[1]):
                rgb = RGBD[r, c, :3]
                xyz = C[:3, r, c]
                if 0 < xyz[2] < 40 :
                    rgb = rgb.astype(str)
                    xyz = xyz.astype(str)
                    file.write("v "+xyz[0]+" "+xyz[1]+" "+xyz[2]+" "+rgb[0]+" "+rgb[1]+" "+rgb[2]+"\n")


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
    # mosaic = np.asarray(Image.open(mosaic_path))
    # a = Image.open(mosaic_path)
    # tiles = image_chunker(mosaic)
    tiles = np.load(mosaic_path)
    for i, tile in enumerate(tiles):
        #np.save(f'E:/tile_{i}.npy', tile)
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

    # create output path if it doesn't exist
    if not os.path.exists(path_out):
        print("Output folder created: /obj")
        os.makedirs(path_out)

    # process texture mosaics
    to_process = [file for file in glob(path_in + "**/generated_textures*.npy")]
    for mosaic_path in tqdm(to_process, desc="Converting PSGAN results into OBJ format"):
        process_mosaic(mosaic_path)

    print("Dataset successfully processed.")
