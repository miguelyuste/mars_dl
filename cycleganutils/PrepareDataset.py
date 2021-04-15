import argparse
import os
import sys
from glob import glob

from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm


def find_files():
    try:
        files_A = (glob(args.path_A + "/**/*.jpg", recursive=True)
                   + glob(args.path_A + "/**/*.png", recursive=True)
                   + glob(args.path_A + "/**/*.gif", recursive=True)
                   + glob(args.path_A + "/**/*.tif", recursive=True))
        files_B = (glob(args.path_B + "/**/*.jpg", recursive=True)
                   + glob(args.path_B + "/**/*.png", recursive=True)
                   + glob(args.path_B + "/**/*.gif", recursive=True)
                   + glob(args.path_B + "/**/*.tif", recursive=True))
    except Exception as e:
        print("\nException while searching for images in given directories: " + repr(e))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        sys.exit(1)
    # the size of the training set is limited by zip() to the length of the shortest of the file lists
    return list(enumerate(zip(files_A, files_B)))


def resize_image(image):
    if image.size[0] > args.resolution and image.size[1] > args.resolution:
        image.resize((args.resolution, args.resolution))
    elif True:
        a = 1


def process_image_pair(paths, pair, file_list_len):
    try:
        index = pair[0]
        im_a = Image.open(pair[1][0])
        try:
            im_a.load()
        except Exception as e:
            print("\nCouldn't load image: " + pair[1][0] + "\nNested exception is: " + repr(e))
            return
        im_b = Image.open(pair[1][1])
        try:
            im_b.load()
        except Exception as e:
            print("\nCouldn't load image: " + pair[1][1] + "\nNested exception is: " + repr(e))
            return
        # we discard the images if either isn't big enough
        # Note: both networks require square images, so param "resolution" is used for both dimensions
        # if im_a.size[0] >= args.resolution and im_a.size[1] >= args.resolution:
        # if im_b.size[0] >= args.resolution and im_b.size[1] >= args.resolution:
        if args.arch == "pix2pix":
            # adjust resolution of both
            if not im_a.size == (args.resolution, args.resolution):
                im_a = im_a.resize((args.resolution, args.resolution))
            # adjust resolution of both
            if not im_b.size == (args.resolution, args.resolution):
                im_b = im_b.resize((args.resolution, args.resolution))
        if index < file_list_len * 4 / 5:
            im_a.save(paths['path_A_train'] + str(index) + ".png")
            im_b.save(paths['path_B_train'] + str(index) + ".png")
            # copy(pair[1][0], paths['path_A_train'] + str(index) + ".png")
            # copy(pair[1][1], paths['path_B_train'] + str(index) + ".png")
        else:
            im_a.save(paths['path_A_test'] + str(index) + ".png")
            im_b.save(paths['path_B_test'] + str(index) + ".png")
            # copy(pair[1][0], paths['path_A_test'] + str(index) + ".png")
            # copy(pair[1][1], paths['path_B_test'] + str(index) + ".png")
    except Exception as e:
        print("\nException while processing following image pairs: " + str(pair) + "\n" + repr(e)) 
        #print("given paths"+str(paths))
        #print("given pair:"+str(pair))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return


def prepare_dataset():
    # create necessary folder structure
    print("Creating folder structure...")
    if args.arch == "pix2pix":
        path_A_train = str(args.output_path) + "/A/train/"
        path_A_test = str(args.output_path) + "/A/test/"
        path_B_train = str(args.output_path) + "/B/train/"
        path_B_test = str(args.output_path) + "/B/test/"
    elif args.arch == "cyclegan":
        path_A_train = str(args.output_path) + "/trainA/"
        path_B_train = str(args.output_path) + "/trainB/"
        path_A_test = str(args.output_path) + "/testA/"
        path_B_test = str(args.output_path) + "/testB/"
    os.makedirs(path_A_train, exist_ok=True)
    os.makedirs(path_B_train, exist_ok=True)
    os.makedirs(path_A_test, exist_ok=True)
    os.makedirs(path_B_test, exist_ok=True)
    # convenience dictionary for parallel function
    paths = {'path_A_train': path_A_train,
             'path_B_train': path_B_train,
             'path_A_test': path_A_test,
             'path_B_test': path_B_test}
    # gather list of files
    to_process = find_files()
    # concurrently process images
    try:
        Parallel(n_jobs=-1, backend="loky")(
            delayed(process_image_pair)(paths=paths, pair=pair, file_list_len=len(to_process)) for pair in
            tqdm(to_process, desc="Processing images"))
    except Exception as e:
        print("Exception while concurrently processing image pairs: " + repr(e))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("arch", help="Architecture to use", choices=['pix2pix', 'cyclegan'])
    parser.add_argument("path_A", help="Path to dataset A")
    parser.add_argument("path_B", help="Path to dataset B")
    parser.add_argument("output_path", help="Path to output folder")
    parser.add_argument("-r", "--resolution", type=int, default=256, help="Resolution to filter out and rescale input "
                                                                          "images. Set by default to the CycleGAN default"
                                                                          " of 256 pixels.")
    args = parser.parse_args()
    if args.arch == "cyclegan":
        print("Creating non-aligned CycleGAN dataset...")
    elif args.arch == "pix2pix":
        print("Creating aligned Pix2Pix dataset...")

    prepare_dataset()
