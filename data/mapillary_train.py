import os
import shutil
from glob import glob
from tqdm import tqdm
from os.path import join
from torch.utils.data import Dataset

DATASET_ROOT = r'/media/cartolab/DataDisk/wuqilong_file/VPR_datasets/MSLS_Dataset/MSLS/'
# This dictionary is copied from the original code
# https://github.com/mapillary/mapillary_sls/blob/master/mapillary_sls/datasets/msls.py#L16
default_cities = {
    'train': ["trondheim", "london", "boston", "melbourne", "amsterdam", "helsinki",
              "tokyo", "toronto", "saopaulo", "moscow", "zurich", "paris", "bangkok",
              "budapest", "austin", "berlin", "ottawa", "phoenix", "goa", "amman", "nairobi", "manila"],
    'val': ["cph", "sf"],
    'test': ["miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala"]
}

csv_files_paths = sorted(glob(join(DATASET_ROOT, "*", "*", "*", "postprocessed.csv"),
                              recursive=True))

print(len(csv_files_paths))


class MapillaryDataset(Dataset):
    def __init__(self, input_transform=None):
        self.input_transform = input_transform

    def __getitem__(self, item):
        pass

    def __len__(self):
        return
