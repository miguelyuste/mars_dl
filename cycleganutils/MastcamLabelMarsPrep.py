path_train_labelMars = "E:\\MFA\\pytorch-CycleGAN-and-pix2pix\\datasets\\PRo3d-Label-Mars-CycleGAN\\trainA\\"
# path_test_labelMars = "E:\\MFA\\pytorch-CycleGAN-and-pix2pix\\datasets\\PRo3d-Label-Mars-CycleGAN\\testA"
path_train_MSLMST = "E:\\MFA\\pytorch-CycleGAN-and-pix2pix\\datasets\\MSLMST_CycleGAN\\trainA\\"
# path_test_MSLMST = ""
path_out_A = "E:\MFA\pytorch-CycleGAN-and-pix2pix\datasets\MSLMST_LabelMars\trainA"
path_out_B = "E:\MFA\pytorch-CycleGAN-and-pix2pix\datasets\MSLMST_LabelMars\trainB"

from tqdm import tqdm
from glob import glob
import os
from PIL import Image

if __name__ == "main":
    processed = 0
    for filename in tqdm(os.listdir(path_train_labelMars),
                         desc='Preparing dataset:'):
        with Image.open(path_train_labelMars + filename, 'r') as label_mars_instance, Image.open(
                path_train_MSLMST + filename, 'r') as mslmst_instance:
            label_mars_instance.save(path_out_B)
            mslmst_instance.save(path_out_B)
            processed += 1
