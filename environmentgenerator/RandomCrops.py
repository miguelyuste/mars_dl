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
from skimage.util.shape import view_as_windows


if __name__ == '__main__':
    print("Initialising random cropper...")

    # load script config
    try:
        with open("config.yaml") as f:
            config = load(f, Loader=FullLoader)
        file_in = config['random_crops']['file_in']
        path_out = config['random_crops']['path_out'] + "/" + "crops/"
        initial_crop_size = config['random_crops']['initial_crop_size']
        random_crop_size = config['random_crops']['random_crop_size']
        no_crops = config['random_crops']['no_crops']
    except Exception as e:
        print("Couldn't load config file: ")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e, exc_type, fname, exc_tb.tb_lineno)
        sys.exit(1)

    # create output path if it doesn't exist
    if not os.path.exists(path_out):
        print("Output folder created: /crops")
        os.makedirs(path_out)

    # load input file
    try:
        input_file = np.load(file_in)
    except Exception as e:
        print("Couldn't load input file: ")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e, exc_type, fname, exc_tb.tb_lineno)
        sys.exit(1)

    try:
        # do initial crop
        initial_crop = input_file[:initial_crop_size,:initial_crop_size, :]

        # do random crops
        for i in range(no_crops):
            idx_x = np.random.randint(0, initial_crop_size-random_crop_size-1)
            idx_y = np.random.randint(0, initial_crop_size-random_crop_size-1)
            crop = initial_crop[idx_x:(idx_x+random_crop_size), idx_y:(idx_y+random_crop_size), :]
            #with open (f'{path_out}/crop_{i}.npy', 'w') as outfile:
            np.save(f'{path_out}/crop_{i}.npy', crop)
        print("All crops generated.")
    except Exception as e:
        print("Exception while extracting crops: ")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e, exc_type, fname, exc_tb.tb_lineno)