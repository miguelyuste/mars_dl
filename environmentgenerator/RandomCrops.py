import os
from yaml import load, FullLoader
import numpy as np
import sys
import traceback


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
        max_empty = config['random_crops']['max_empty']
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
        valid_crops = 0
        # do initial crop if a crop size was given
        if initial_crop_size > 0:
            initial_crop = input_file[:initial_crop_size,:initial_crop_size, :]
            print(initial_crop[:,:,3].max(), initial_crop[:,:,3].min())
            initial_crop[:, :, 3] = (initial_crop[:,:,3]-initial_crop[:,:,3].min())/(initial_crop[:,:,3].max() - initial_crop[:,:,3].min()) - 0.5
        else:
            initial_crop = input_file
            print(initial_crop[:,:,3].max(), initial_crop[:,:,3].min())
            initial_crop[:, :, 3] = (initial_crop[:,:,3]-initial_crop[:,:,3].min())/(initial_crop[:,:,3].max() - initial_crop[:,:,3].min()) - 0.5
        # clip RGB values greater than 1 and then normalise to {-1,1}
        initial_crop[:, :, :3] = np.clip(initial_crop[:, :, :3], 0, 1) * 2 - 1
        # do random crops
        print("Extracting random crops...")
        for i in range(no_crops*1000):
            idx_x = np.random.randint(0, initial_crop.shape[0]-random_crop_size-1)
            idx_y = np.random.randint(0, initial_crop.shape[1]-random_crop_size-1)
            crop = initial_crop[idx_x:(idx_x+random_crop_size), idx_y:(idx_y+random_crop_size), :]
            # calculate how many pixels are empty
            empty_pixel_count = (crop[:, :, 3] > 40).sum()
            total_pixel_count = crop.shape[0] * crop.shape[1]
            percentage_empty = 100 * (empty_pixel_count / total_pixel_count)
            # discard if a big part of the tile is empty
            if percentage_empty <= max_empty:
                # discard if the texture is not colourful enough
                if np.mean(crop[:, :, :3] * 255).astype(int) > 100:
                    np.save(f'{path_out}/crop_{i}.npy', crop)
                    valid_crops += 1
            # if we have enough valid crops, stop
            if valid_crops == no_crops:
                break
        print(f"{valid_crops} crops generated.")
    except Exception as e:
        print("Exception while extracting crops: ")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e, exc_type, fname, exc_tb.tb_lineno)
        traceback.print_exc()