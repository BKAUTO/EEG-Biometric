import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class CP_MixedNet(nn.Module):
    def __init__(self):
        super(CP_MixedNet, self).__init__()

        # ***CP-Spatio-Temporal Block***
        # Channel Projection
        self.channelProj = nn.Conv2d(22, 35, 1, stride=1, bias=False)   # (22*1*1125)->(35*1*1125)
        self.batchnorm_proj_tranf = nn.BatchNorm2d(35)
        # Shape Transformation
        self.shapeTrans = nn.Conv2d(35, 35, 1, stride=1, bias=False)    # (35*1*1125)->(35*1*1125)
        # Temporal Convolution
        self.drop1 = nn.Dropout2d(p=0.5)
        self.conv1 = nn.Conv2d(1, 25, (1,11), stride=1, bias=False)     # (1*35*1125)->(25*35*1115)
        self.batchnorm1 = nn.BatchNorm2d(25, False)
        # Spatial Convolution
        self.drop2 = nn.Dropout2d(p=0.5)
        self.conv2 = nn.Conv2d(25, 25, (35,1), stride=1, bias=False)    # (25*35*1115)->(25*1*1115)
        self.batchnorm2 = nn.BatchNorm2d(25, False)
        # Max Pooling
        self.maxPool1 = nn.MaxPool2d((1,3), stride=3, padding=0)    # (25*1*1115)->(25*1*371)

        # ***MS-Conv Block***
        # unDilated Convolution
        self.drop3 = nn.Dropout2d(p=0.5)
        self.conv3 = nn.Conv2d(25, 100, 1, stride=1, bias=False)        # (25*1*371)->(100*1*371)
        self.batchnorm3 = nn.BatchNorm2d(100)
        self.drop4 = nn.Dropout2d(p=0.5)
        self.conv4 = nn.Conv2d(100, 100, (1, 11), stride=1, padding=(0,5), bias=False)  # (100*1*371)->(100*1*371)
        self.batchnorm4 = nn.BatchNorm2d(100)
        # Dilated Convolution
        self.dropDil = nn.Dropout2d(p=0.5)
        self.dilatedconv = nn.Conv2d(100, 100, (1,11), stride=1, padding=(0,10), dilation=2, bias=False)    # (100*1*371)->(100*1*371)
        self.batchnormDil = nn.BatchNorm2d(100)
        # Max pooling after Concatenating
        self.batchnorm_cancat = nn.BatchNorm2d(225)
        self.poolConcatenated = nn.MaxPool2d((1,3), stride=3, padding=0)    # (225*1*371)->(225*1*123)


        # ***Classification Block***
        self.drop5 = nn.Dropout(p=0.5)
        self.conv5 = nn.Conv2d(225, 225, (1,11), stride=1)  # (225*1*123)->(225*1*113)
        self.batchnorm5 = nn.BatchNorm2d(225)
        self.maxPool2 = nn.MaxPool2d((1,3), stride=3, padding=0)    # (225*1*113)->(225*1*37)
        self.fc = nn.Linear(8325, 2, bias=False)    # (1*8325)->(1*4)
        # self.softmax = nn.Softmax(dim=1)
        self.batchnorm6 = nn.BatchNorm1d(2)
        self.softmax = nn.LogSoftmax(dim=1)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = torch.unsqueeze(x, 2)
        x = F.elu(self.batchnorm_proj_tranf(self.channelProj(x)))
        # print('Channel Projection:',x)
        x = F.elu(self.batchnorm_proj_tranf(self.shapeTrans(x)))
        #print('before Shape Transformation:',x.shape)
        x = torch.transpose(x, 1, 2)
        #print('after Shape Transformation:',x.shape)
        x = F.elu(self.batchnorm1(self.conv1(self.drop1(x))))
        #print('Temporal convolution:',x.shape)
        x = F.elu(self.batchnorm2(self.conv2(self.drop2(x))))
        #print('Spatial convolution:',x.shape)
        x = self.maxPool1(x)
        # print('Max pooling：',x.shape)

        x1 = F.elu(self.batchnorm3(self.conv3(self.drop3(x))))
        x_dilated = F.elu(self.batchnormDil(self.dilatedconv(self.dropDil(x1))))
        #print('Dilated Convolution1:', x_dilated.shape)
        x_undilated = F.elu(self.batchnorm4(self.conv4(self.drop4(x1))))
        #print('Undilated Convolution2:', x_undilated.shape)

        x = torch.cat((x, x_dilated, x_undilated),dim=1)
        # print('Concatenated:', x.shape)
        x = self.poolConcatenated(self.batchnorm_cancat(x))
        #print('MixedScaleConv:', x.shape)

        x = F.elu(self.batchnorm5(self.conv5(self.drop5(x))))
        #print('Conv5:', x.shape)
        x = self.maxPool2(x)
        # print('maxPool2:', x.shape)
        x = x.view(-1, 8325)
        # print('beforeFC:', x.shape)
        x = F.relu(self.batchnorm6(self.fc(x)))
        #print('FC:', x.shape)
        x = self.softmax(x)
        #print('softmax:', x.shape)
        return x