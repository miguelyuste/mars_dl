from bs4 import BeautifulSoup
from tqdm import tqdm
import os

from PIL import Image

# Todo: move these to config
path_in = "D:\\Datasets\\Label-Mars\\NOAH-SSL-DB-001-DB1 Annotations and Images\\"
path_out = "D:\\Datasets\\Label-Mars\\NOAH-SSL-DB-001-DB1 Annotations and Images\\images_clean"


def filter_artificial(path_in, path_out):
    processed = 0
    for filename in tqdm(os.listdir(path_in + "/annotations"), desc='Filtering out pictures containing artificial objects:'):
        with open(path_in + "/annotations/" + filename, 'r') as annotations:
            data = annotations.read()
            soup = BeautifulSoup(data, 'xml')
            # Todo: filter out tracks too?
            if not any(tag.contents[0].startswith("Artificial") for tag in soup.select('name')):
                image = Image.open(path_in + "images/" + filename.strip(".xml") + ".jpg")
                copy_out_path = path_out + "/" + filename.strip(".xml")
                if os.path.exists(f"{copy_out_path}.png"):
                    i = 2
                    while os.path.exists("{}_{}.png".format(copy_out_path, i)):
                        i += 1
                    copy_out_path += "_" + str(i)
                copy_out_path += ".png"
                image.save(copy_out_path)
                processed += 1
    return processed


if __name__ == '__main__':
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    print("Total number clean pictures: {}".format(filter_artificial(path_in, path_out)))
