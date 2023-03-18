# @article{das2018a,
#   author = {Das, Rig and Maiorana, Emanuele and Campisi, Patrizio},
#   title = {MOTOR IMAGERY FOR EEG BIOMETRICS USING CONVOLUTIONAL NEURAL NETWORK},
#   language = {eng},
#   format = {article},
#   journal = {2018 Ieee International Conference on Acoustics, Speech and Signal Processing (icassp)},
#   pages = {2062-2066},
#   year = {2018},
#   isbn = {9781538646588},
#   publisher = {IEEE}
# }
import torch 
import torch.nn as nn

class MI_CNN(nn.Module):
    def __init__(self, nChan):
        super(MI_CNN, self).__init__()

        self.layer1 = nn.Conv2d(1, 640, (5,5))
        self.max_pool1 = nn.MaxPool2d((2, 2))
        self.layer2 = nn.Conv2d(640, 512, (5,5))
        self.max_pool2 = nn.MaxPool2d((2, 2))
        self.layer3 = nn.Conv2d(512, 1024, (1,157))
        self.relu = nn.ReLU()
        self.layer4 = nn.Conv2d(1024, 2, (1,1))
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        # x = torch.squeeze(x)
        x = torch.unsqueeze(x,1)
        x = self.layer1(x)
        x = self.max_pool1(x)
        x = self.layer2(x)
        x = self.max_pool2(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = torch.squeeze(x)
        x = self.softmax(x)
        # x = torch.unsqueeze(x,0)
        # print(x.shape)
        return x
