import os
from tqdm import tqdm
from shutil import copy

path_A = "D:/Datasets/Label-Mars/NOAH-SSL-DB-001-DB1 Annotations and Images/images_clean"
path_B = "D:/PRo3D_Snapshots/preprocessed"
path_out = "D:/Datasets/Label-Mars-CycleGAN"

if __name__ == '__main__':
    if not os.path.exists(path_out+"/trainA"):
        os.makedirs(path_out+"/trainA")
    if not os.path.exists(path_out+"/trainB"):
        os.makedirs(path_out+"/trainB")
    if not os.path.exists(path_out+"/testA"):
        os.makedirs(path_out+"/testA")
    if not os.path.exists(path_out+"/testB"):
        os.makedirs(path_out+"/testB")
    files_A = os.listdir(path_A)
    files_B = os.listdir(path_B)
    for i, (a, b) in enumerate(tqdm(zip(files_A, files_B))):
        if i < len(files_A)*4/5:
            copy(path_A + "/" + a, path_out+"/trainA/"+str(i)+".png")
            copy(path_B + "/" + b, path_out+"/trainB/"+str(i)+".png")
        else:
            copy(path_A + "/" + a, path_out+"/testA/"+str(i)+".png")
            copy(path_B + "/" + b, path_out+"/testB/"+str(i)+".png")