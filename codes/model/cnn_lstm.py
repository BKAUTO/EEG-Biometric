import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()

        self.layer1 = nn.Conv1d(64, 128, 2)
        self.relu = nn.ReLU()
        self.layer2 = nn.Conv1d(128, 256, 2)
        self.layer3 = nn.Conv1d(256, 512, 2)
        self.layer4 = nn.Conv1d(512, 1024, 2)

        self.layer5 = nn.Linear(636*1024, 192)
        self.dropout = nn.Dropout(p=0.5)
        self.layer6 = nn.LSTM(1, 1, batch_first=True)
        self.layer7 = nn.LSTM(1, 1, batch_first=True)
        self.layer8 = nn.Linear(192, 192)
        self.layer9 = nn.Linear(192, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.layer5(x)
        x = self.dropout(x)
        x = torch.unsqueeze(x,2)
        x, (h_n, c_n) = self.layer6(x)
        x, (h_n, c_n) = self.layer7(x)
        x = torch.flatten(x, start_dim=1)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.softmax(x)
        return x