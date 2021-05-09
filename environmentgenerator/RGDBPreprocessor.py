import os
from glob import glob

from joblib import Parallel, delayed
from tqdm import tqdm
from yaml import load, FullLoader
import numpy as np
from PIL import Image
from pathlib import Path


# ToDo: conversion to actual RGB values

def chunker(image):
    # max number of chunks per axis, last chunk on each axis will overlap slightly
    chunks_x = int(np.ceil(image.shape[0] / chunk_resolution[0]))
    chunks_y = int(np.ceil(image.shape[1] / chunk_resolution[1]))
    chunks = []
    # get indices of chunk, then slice chunk and append
    # todo: RANGES MIGHT BE WRONG! check objconverter
    for i in range(chunks_x - 1):
        for j in range(chunks_y - 1):
            if i != (chunks_x - 1):  # iterate ascendingly in steps of size chunk_resolution[x]
                idx_x = [chunk_resolution[0] * i, chunk_resolution[0] * (i + 1)]
            else:  # overlap
                idx_x = [image.shape[0] - chunk_resolution[0], image.shape[0]]
            if j != (chunks_y - 1):  # iterate ascendingly in steps of size chunk_resolution[y]
                idx_y = [chunk_resolution[1] * j, chunk_resolution[1] * (j + 1)]
            else:  # overlap
                idx_y = [image.shape[1] - chunk_resolution[1], image.shape[1]]
            chunks.append(image[idx_x[0]:idx_x[1], idx_y[0]:idx_y[1]])
    return chunks


def preprocess_rbgd(image_filepath):
    # load .npy file containing RGBD image
    #image = Image.open(image_filepath)
    #image.load()
    image = np.load(image_filepath)
    rgbd_image = np.asarray(image)
    # get chunks, save only if % of "empty" pixels is lower than set threshold
    for i, chunk in enumerate(chunker(rgbd_image)):
        empty_pixel_count = (chunk[:, :, 3] > 40).sum()
        total_pixel_count = chunk.shape[0] * chunk.shape[1]
        percentage_empty = 100 * (empty_pixel_count / total_pixel_count)
        if percentage_empty < config['max_empty']:
            #outfile = Image.fromarray(chunk, 'RGBA')
            #outfile.save(path_out + "/" + Path(image_filepath).stem + f"_{i}.png")
            np.save(str(path_out + "/" + Path(image_filepath).stem + f"_{i}.npy"),chunk.astype(np.float32))


if __name__ == '__main__':
    print("Initialising RGBD image preprocessor...")

    # load script config
    with open("config.yaml") as f:
        config = load(f, Loader=FullLoader)
    config = config['psganpreprocess']
    path_in = config['path_in']
    path_out = path_in + config['path_out'] + "_empty=" + str(config['max_empty'])
    chunk_resolution = config['chunk_res']

    # create output path if it doesn't exist
    if not os.path.exists(path_out):
        print(f"Output folder created: {config['path_out']}")
        os.makedirs(path_out)

    # concurrently process images
    Parallel(n_jobs=-1, backend="loky")(
        map(delayed(preprocess_rbgd), (file for file in
                                       tqdm(glob(path_in + "/*.npy"),
                                            desc="Preprocessing RGBD images"))))
    print("Dataset successfully processed.")
