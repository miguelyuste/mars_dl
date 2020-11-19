# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:19:37 2020

@author: migue
"""

import argparse
import os
import subprocess
from glob import glob
from pathlib import Path

from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm


def run_adjust(file):
    # convert infile to JPG
    infile_jpg = Path(file).with_suffix(".jpg")
    image = Image.open(file).convert("RGB")
    image.save(Path(file).with_suffix(".jpg"))
    # run script on JPG infile
    outfile_jpg = args.output + "/" + Path(file).stem + "_corrected.jpg"
    subprocess.run(["HistAdopt.exe", "Ref.jpg", f"{infile_jpg}", f"{outfile_jpg}"])
    # delete JPG infile
    os.remove(infile_jpg)
    # convert outfile to PNG
    outfile_png = Path(outfile_jpg).with_suffix(".png")
    image = Image.open(outfile_jpg).convert("RGB")
    image.save(outfile_png)
    # delete JPG outfile
    os.remove(outfile_jpg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("-p", "--path", help="Path to script", required=True)
    parser.add_argument("-i", "--input",
                        help="path to input folder", required=True)
    # parser.add_argument("-r", "--reference_image",
    #                        help="path to reference image", required=True)
    parser.add_argument("-o", "--output",
                        help="path to folder where to place the processed images", required=True)

    args = parser.parse_args()

    files = [file for file in glob(args.input + '/**/*.png', recursive=True) if
             "mask" not in file and "depth" not in file]

    Parallel(n_jobs=16, backend="loky")(
        map(delayed(run_adjust), (file for file in tqdm(files, desc="Processing images"))))

    print("All images processed successfully.")
