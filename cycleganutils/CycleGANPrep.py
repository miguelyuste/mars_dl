import os
from tqdm import tqdm
from shutil import copy
import yaml

if __name__ == '__main__':
    # load object config file
    with open('snapshotutils/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if not os.path.exists(config['path_out']+"/trainA"):
        os.makedirs(config['path_out']+"/trainA")
    if not os.path.exists(config['path_out']+"/trainB"):
        os.makedirs(config['path_out']+"/trainB")
    if not os.path.exists(config['path_out']+"/testA"):
        os.makedirs(config['path_out']+"/testA")
    if not os.path.exists(config['path_out']+"/testB"):
        os.makedirs(config['path_out']+"/testB")
    files_A = os.listdir(config['path_A'])
    files_B = os.listdir(config['path_B'])
    for i, (a, b) in enumerate(tqdm(zip(files_A, files_B))):
        if i < len(files_A)*4/5:
            copy(config['path_A'] + "/" + a, config['path_out']+"/trainA/"+str(i)+".png")
            copy(config['path_B'] + "/" + b, config['path_out']+"/trainB/"+str(i)+".png")
        else:
            copy(config['path_A'] + "/" + a, config['path_out']+"/testA/"+str(i)+".png")
            copy(config['path_B'] + "/" + b, config['path_out']+"/testB/"+str(i)+".png")