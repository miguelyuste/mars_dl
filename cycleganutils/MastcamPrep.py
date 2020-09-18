from PIL import Image

from yaml import load, FullLoader
from glob import glob
import os
from sys import exit
from tqdm import tqdm
import concurrent

def adjust_resolution(im, resolution):
    # if resolution not correct, adjust it
    if not im.size == resolution:
        im.thumbnail(resolution)


def preprocess_image(file):
    im = Image.open(file)
    # we discard the image if it isn't big enough
    if im.size[0] >= resolution[0] and im.size[1] >= resolution[1]:
        # adjust resolution
        adjust_resolution(im, resolution)
        # save RGB image to train_A and grayscale to train_B
        im.save(path_A + file.strip(os.path.split(file)[0]).strip('.tif') + ".png")
        im.convert('LA').save(path_B + file.strip(os.path.split(file)[0]).strip('.tif') + ".png")


if __name__ == '__main__':
    # load object config file
    print("Loading configuration...")
    with open('./config.yaml') as f:
        config = load(f, Loader=FullLoader)
    if not config['mastcam_pathIn']:
        print("Input path configuration missing. Please specify the input path in the YAML config file.")
        exit(1)
    elif not config['mastcam_pathOut']:
        print("Output path configuration missing. Please specify the output path in the YAML config file.")
        exit(1)
    # create output folders if they doesn't exist
    if not os.path.exists(config['mastcam_pathOut']):
        os.makedirs(config['mastcam_pathOut'])
    path_A = config['mastcam_pathOut']+"/trainA/"
    if not os.path.exists(path_A):
        os.makedirs(path_A)
    path_B =config['mastcam_pathOut']+"/trainB/"
    if not os.path.exists(path_B):
        os.makedirs(path_B)
    # config resolution values are read as string
    resolution = tuple(config['resolution'])
    # concurrently process images
    executor = concurrent.futures.ProcessPoolExecutor(config['mastcam_processPool'])
    futures = [executor.submit(preprocess_image, file) for file in tqdm(glob(config['mastcam_pathIn'] + "**/*.tif", recursive=True), desc="Processing images:")]
    concurrent.futures.wait(futures)
    print("Dataset successfully processed.")
