from yaml import load, FullLoader
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from csv import writer
from pathlib import Path


def calculate_sigma(file):
    # calculate standard deviation ignoring OBJ points not representing empty space
    depth = np.load(file)[:, :, 3]
    sigma = np.std(depth[depth < 40])
    return [file, sigma]


if __name__ == '__main__':
    print("Initialising standard deviation calculator...")

    # load script config
    try:
        with open("C:\\Users\\migue\\PycharmProjects\\mars_dl\\environmentgenerator\\config.yaml") as f:
            config = load(f, Loader=FullLoader)
        path_in = Path(config['sigma_calculator']['path_in'])
        path_out = config['sigma_calculator']['path_out']
    except Exception as e:
        print("Exception while loading config: " + repr(e))

    try:
        # non-recursive search to avoid catching preprocessed npy files
        to_process = [file for file in path_in.glob("*.npy")]
        # concurrently calculate sigmas
        sigmas = Parallel(n_jobs=-1, backend="loky")(
            map(delayed(calculate_sigma),
                (mosaic_path for mosaic_path in tqdm(to_process, desc="Calculating individual sigmas"))))
        average_sigma = np.average([sigma[1] for sigma in sigmas])
        print("Average standard deviation: " + str(average_sigma))
        # write individual results to csv
        with open(path_out + '/sigmas.csv', 'w', newline='') as outfile:
            writer = writer(outfile)
            writer.writerow(["OBJ", "Standard Deviation"])
            writer.writerow(["AVERAGE", str(average_sigma)])
            writer.writerows(sigmas)
            print("Results written to outfile: " + path_out + 'sigmas.csv')
    except Exception as e:
        print("Exception while calculating individual standard deviations: " + repr(e))
