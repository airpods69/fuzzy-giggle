import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import mne

data_path = "/mnt/storage/Capstone/Dataset/BCICIV_2a_gdf/A01T.gdf"
data = mne.io.read_raw_gdf(data_path, preload=True)
data.filter(4, 40, method='iir')

raw_eeg = data.get_data()
events_annotations = mne.events_from_annotations(data)

def label_eeg(raw_data: np.ndarray, labels: np.ndarray):
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

labeled_eeg = np.array(label_eeg(raw_eeg.T, events_annotations), dtype=object)


def eeg_split_freq(labeled_eeg: np.ndarray):
    # splits the labeled eeg signals in batches of 250 (sampling rate)
    split = []
    for eeg_label in labeled_eeg:
        for i in range(0, len(eeg_label[0].T) - 250, 250):
            split.append([eeg_label[0].T[i: i + 250].T, eeg_label[1]])

    return split

eeg_split = eeg_split_freq(labeled_eeg)
eeg_split.pop()
eeg_split.pop()

class EEG_DATASET(Dataset):
    def __init__(self, eeg_data: np.ndarray):
        self.eeg_data = eeg_data

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        return torch.tensor(np.array([self.eeg_data[idx][0].T]).astype('float32')), torch.tensor(np.array(self.eeg_data[idx][1]).astype('float32'))

eeg_dataset = EEG_DATASET(eeg_split)
print(eeg_dataset.__getitem__(0)[0].shape)

eeg_loader = DataLoader(eeg_dataset, batch_size = 16, shuffle = True)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class EEGNET(nn.Module):
    def __init__(self, receptive_field=250, mean_pool=5, filter_sizing = 2, dropout = 0.2, D = 1):
        super(EEGNET,self).__init__()
        channel_amount = 22
        num_classes = 2
        self.temporal=nn.Sequential(
            nn.Conv2d(1,filter_sizing,kernel_size=[1,receptive_field],stride=1, bias=False,\
                padding='same'), 
            nn.BatchNorm2d(filter_sizing),
        )
        self.spatial=nn.Sequential(
            nn.Conv2d(filter_sizing,filter_sizing*D,kernel_size=[channel_amount,1],bias=False,\
                groups=filter_sizing),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ELU(True),
        )

        self.seperable=nn.Sequential(
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=[1,16],\
                padding='same',groups=filter_sizing*D, bias=False),
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=[1,1], padding='same',groups=1, bias=False),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ELU(True),
        )
        self.avgpool1 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)   
        self.avgpool2 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)
        self.dropout = nn.Dropout(dropout)
        self.view = nn.Sequential(Flatten())

        endsize = 1832
        self.fc2 = nn.Linear(endsize, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out = self.temporal(x)
        # print(out.shape)
        out = self.spatial(out)
        # print(out.shape)
        out = self.avgpool1(out)
        # print(out.shape)
        out = self.dropout(out)
        # print(out.shape)
        out = self.seperable(out)
        # print(out.shape)
        # out = self.avgpool2(out)
        # print(out.shape)
        out = self.dropout(out)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        prediction = self.fc2(out)
        return self.sigmoid(prediction)


model = EEGNET()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

from torchsummary import summary
summary(EEGNET().to(device), (1, 250, 22))


def evaluation():
    model.eval()
    

epochs = 100
losses = []
accuracy = []
for epoch in range(epochs):
    loss = []
    print(f'Epoch: {epoch+1}')

    train_acc = 0

    for _, (data, label) in tqdm(enumerate(eeg_loader)):
        optimizer.zero_grad()
        data = data.to(device)
        if len(label) == 10:
            continue
        label = label.reshape((16,1)).to(device)
        y_pred = model(data)
        batch_loss = criterion(y_pred,label)
        train_acc += torch.sum(y_pred.round() == label)

        loss.append(batch_loss.detach().cpu().numpy())
        batch_loss.backward()
        optimizer.step()
    
    print(f'Training loss = {np.mean(loss)}, Training Accuracy = {100 * train_acc / len(eeg_dataset)}')
    losses.append(np.mean(loss))
    accuracy.append(100 * train_acc.detach().cpu().numpy() / len(eeg_dataset))


plt.subplot(1,2,1)
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(accuracy)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
