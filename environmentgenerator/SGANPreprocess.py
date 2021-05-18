import os
import sys
import traceback
from glob import glob
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from yaml import load, FullLoader


def random_crops(image_filepath):
    v_parameters = []
    crop_filepaths = []
    try:
        # load .npy file containing RGBD image
        image = np.load(image_filepath)
        # we'll count the number of valid crops we get and store their vmax and vmin
        valid_crops = 0
        # do until we have enough crops or we've made (10 * target number of crops) attempts
        for i in range(no_crops * 10):
            # random crop of image
            idx_x = np.random.randint(0, image.shape[0] - random_crop_size - 1)
            idx_y = np.random.randint(0, image.shape[1] - random_crop_size - 1)
            crop = image[idx_x:(idx_x + random_crop_size), idx_y:(idx_y + random_crop_size), :]
            # calculate how many pixels are empty
            empty_pixel_count = (crop[:, :, 3] > 40).sum()
            total_pixel_count = crop.shape[0] * crop.shape[1]
            percentage_empty = 100 * (empty_pixel_count / total_pixel_count)
            # discard if a big part of the tile is empty
            if percentage_empty <= config['max_empty']:
                # discard if the texture is not colourful enough
                if np.mean(crop[:, :, :3] * 255).astype(int) > 100:
                    # save crop
                    crop_filepath = str(path_out + "/" + Path(image_filepath).stem + f"_{i}.npy")
                    np.save(crop_filepath, crop.astype(np.float32))
                    # we store the vmax and vmin of the crop to calculate the overall average vmax and vmin later on
                    v_parameters.append([crop[:, :, 3].max(), crop[:, :, 3].min()])
                    # we save the filepath to reload the crop in the normalisation
                    crop_filepaths.append(crop_filepath)
                    # we count this crop
                    valid_crops += 1
            # if we have enough valid crops, stop
            if valid_crops == no_crops:
                break
    except Exception as e:
        print(f"Exception while working with image {str(image_filepath)}: ")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e, exc_type, fname)
        traceback.print_exc()
    return v_parameters, crop_filepaths


def normalise(crop_path):
    try:
        crop = np.load(crop_path)
        # normalise depth values to {-0.5,0.5}
        crop[:, :, 3] = (crop[:, :, 3] - avg_vmin) / (avg_vmax - avg_vmin) - 0.5
        # clip RGB values greater than 1 and then normalise to {-1,1}
        crop[:, :, :3] = np.clip(crop[:, :, :3], 0, 1) * 2 - 1
        # save normalised crop
        np.save(str(Path(crop_path).with_suffix("")) + "_normalised.npy", crop.astype(np.float32))
    except Exception as e:
        print(f"Exception while normalising crop: {str(crop_path)}: ")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e, exc_type, fname)
        traceback.print_exc()


if __name__ == '__main__':
    print("Initialising RGBD image preprocessor...")

    # load script config
    with open("config.yaml") as f:
        config = load(f, Loader=FullLoader)
    config = config['sganpreprocess']
    path_in = config['path_in']
    random_crop_size = config['random_crop_size']
    no_crops = config['no_crops']
    path_out = path_in + config['path_out'] + "_empty=" + str(config['max_empty']) + "_res=" + str(random_crop_size)

    # create output path if it doesn't exist
    if not os.path.exists(path_out):
        print(f"Output folder created: {path_out}")
        os.makedirs(path_out)

    # concurrently process images
    try:
        # extract and save random crops according to filter conditions
        crops_and_vs = Parallel(n_jobs=-1, backend="loky")(
            map(delayed(random_crops), (file for file in
                                        tqdm(glob(path_in + "/*.npy"),
                                             desc="Extracting crops of RGBD images"))))
        all_v_parameters = np.asarray(sum((e[0] for e in crops_and_vs), []))
        crop_filepaths = sum((e[1] for e in crops_and_vs), [])
        # calculate average max and min of crops
        avg_vmax = np.average(all_v_parameters[:, 0])
        avg_vmin = np.average(all_v_parameters[:, 1])
        print(f"Average parameters of crop dataset: vmax={avg_vmax}, vmin={avg_vmin}")
        # save individual vmax and vmin values for later reference
        np.save(str(path_out + "/all_v_params.npy"), all_v_parameters)
        # reload crops and do normalisation
        Parallel(n_jobs=-1, backend="loky")(map(delayed(normalise), (crop_path for crop_path in tqdm(crop_filepaths,
                                                                                                     desc="Normalising crop RGBD values"))))
    except Exception as e:
        print("Exception while extracting and normalising crops: ")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e, exc_type, fname)
        traceback.print_exc()
    print("Dataset processed.")
