import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import mne

class EEG_DATASET(Dataset):
    def __init__(self, data_path: str):
        data = self.get_data(data_path)
        self.eeg_data = self.eeg_split_freq(data)


    def get_data(self, data_path: str):
        data = mne.io.read_raw_gdf(data_path, preload=True)
        data.filter(4, 40, method='iir') # 4 - 40 Hz to get Beta and Mu frequency bandwidths (I think there would be some other band in between but lets ignore that for now)
        raw_eeg = data.get_data()
        events_annotations = mne.events_from_annotations(data)
        labeled_eeg = np.array(self.label_eeg(raw_eeg.T, events_annotations), dtype=object)

        return labeled_eeg

    def label_eeg(self, raw_data: np.ndarray, labels: np.ndarray):
        # splits the eeg_data whenever the data label changes 
        # and gives that segment the label
        annotations = labels[0]
        labeled_eeg = []
        for i in range(len(annotations) - 1):
            # if time is the same, we don't really need it
            if annotations[i][0] == annotations[i + 1][0]:
                continue
            else:
                if annotations[i][-1] not in [1, 2, 3, 4, 5, 6, 9, 10]:
                    eeg_with_label = [raw_data[annotations[i][0]: annotations[i + 1][0]].T[:22], annotations[i][-1] - 7] # removes the eog signals here as well
                    labeled_eeg.append(eeg_with_label)
            labeled_eeg.append([raw_data[annotations[-1][0]:].T[:22], annotations[i][-1] - 7])
        return labeled_eeg


    def eeg_split_freq(self, labeled_eeg: np.ndarray):
        # splits the labeled eeg signals in batches of 250 (sampling rate)
        split = []
        for eeg_label in labeled_eeg:
            for i in range(0, len(eeg_label[0].T) - 250 - 2, 250): # idk why the -2 exists but it was in my previous code so lets not remove it
                split.append([eeg_label[0].T[i: i + 250].T, eeg_label[1]])

        return split

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        return torch.tensor(np.array([self.eeg_data[idx][0].T]).astype('float32')), torch.tensor(np.array(self.eeg_data[idx][1]).astype('float32'))
