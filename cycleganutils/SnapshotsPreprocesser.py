import cv2
import os
from shutil import copy
import argparse
from tqdm import tqdm


def is_empty(image):
    if cv2.countNonZero(image) == 0:
        return True
    else:
        return False

#todo: more than x % is empty

def filter_empty(search_dir, output_dir):
    processed = 0
    for filename in tqdm(os.listdir(search_dir), desc='Filtering empty snapshots:'):
        if filename.endswith(".png"):
            image = cv2.imread(search_dir + "/" + filename, 0)
            if not is_empty(image):
                copy_out_path = output_dir + "/" + str(processed) + ".png"
                if os.path.exists(f"{copy_out_path}.json"):
                    i = 2
                    while os.path.exists("{}_{}.json".format(copy_out_path, i)):
                        i += 1
                    copy_out_path += "_" + str(i)
                # todo: copy corresponding SC/OPC too
                copy(search_dir + "/" + filename, copy_out_path)
                processed += 1
    return processed


if __name__ == '__main__':
    print("Running preprocessing...")
    # optional argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input folder", required=True)
    parser.add_argument("-o", "--output",
                        help="output folder where to place the preprocessed images; input_path/preprocessed by default")
    args = parser.parse_args()
    # params take default values for snapshot generation
    if args.output:
        out_path = args.output
    else:
        out_path = args.input + "/preprocessed"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print("Total number of non-empty pictures found: {}".format(filter_empty(args.input, out_path)))
