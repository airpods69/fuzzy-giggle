import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from model import EEGNET
from dataset import EEG_DATASET

import mne

# so I think I will keep the whole thing in classes instead of functions. Why? I wanna look cool

class Driver:
    def __init__(self, path_to_dataset) -> None:
        self.dataset = EEG_DATASET(path_to_dataset)

        for i in range(self.dataset.__len__()):
            print(self.dataset.__getitem__(i)[1])


Driver("/mnt/storage/Capstone/Dataset/BCICIV_2a_gdf/A01T.gdf")
