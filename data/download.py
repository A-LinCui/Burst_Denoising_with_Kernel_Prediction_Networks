import os
import sys

from tqdm import tqdm
from urllib.request import urlretrieve


if __name__ == "__main__":
    url_file = os.path.join(os.path.dirname(__file__), "filesAdobe.txt")
    
    save_path = os.path.join(os.path.dirname(__file__), "Adobe5K")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    url_lst = []
    with open(url_file, 'r') as f:
        for line in f.readlines():
            url_lst.append(line.rstrip("\n"))

    for url in tqdm(url_lst):
        used_url = "https://data.csail.mit.edu/graphics/fivek/img/dng/" + url + ".dng"
        urlretrieve(used_url, os.path.join(save_path, url + ".dng"))
