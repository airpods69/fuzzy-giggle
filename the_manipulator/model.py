# Architecture of the model is defined here
import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class EEGNET(nn.Module):
    def __init__(self, receptive_field=250, mean_pool=5, filter_sizing = 2, dropout = 0.2, D = 1):
        super(EEGNET,self).__init__()
        channel_amount = 22
        num_classes = 2
        self.temporal=nn.Sequential(
            nn.Conv2d(1,filter_sizing,kernel_size=(1,receptive_field),stride=1, bias=False,\
                padding='same'), 
            nn.BatchNorm2d(filter_sizing),
        )
        self.spatial=nn.Sequential(
            nn.Conv2d(filter_sizing,filter_sizing*D,kernel_size=(channel_amount,1),bias=False,\
                groups=filter_sizing),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ELU(True),
        )

        self.seperable=nn.Sequential(
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=(1,16),\
                padding='same',groups=filter_sizing*D, bias=False),
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=(1,1), padding='same',groups=1, bias=False),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ELU(True),
        )
        self.avgpool1 = nn.AvgPool2d((1, 5), stride=(1, 5), padding=0)   
        self.avgpool2 = nn.AvgPool2d((1, 5), stride=(1, 5), padding=0)
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

