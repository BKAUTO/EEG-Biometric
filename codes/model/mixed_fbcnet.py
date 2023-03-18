import torch
import torch.nn as nn
import sys
import math
current_module = sys.modules[__name__]

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm = True, max_norm=2, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim = self.dim, keepdim=True), 1e-6, 1e6))

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=0.5, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

class MIXED_FBCNet(nn.Module):
    '''
    The data input is in a form of batch x 1 x chan x time x filterBand
    '''
    def SCB(self, m, nChan, nBands, doWeightNorm=True, *args, **kargs):
        '''
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        ''' 
        return nn.Sequential(
                Conv2dWithConstraint(nBands, m*nBands, (nChan, 1), groups=nBands,
                                     max_norm=2 , doWeightNorm=doWeightNorm, padding = 0),
                nn.BatchNorm2d(m*nBands),
                swish()
                )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                LinearWithConstraint(inF, outF, max_norm = 0.5, doWeightNorm = doWeightNorm, *args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def ChannelProj(self, nChan, nProjChan, doWeightNorm=True):
        return nn.Sequential(
            Conv2dWithConstraint(nChan, nProjChan, (1,1), max_norm=1, doWeightNorm=doWeightNorm, padding=0, bias=False),
            nn.BatchNorm2d(nProjChan),
            swish()
        )
    
    def ShapeTrans(self, nProjChan1, nProjChan2, doWeightNorm=True):
        return nn.Sequential(
            Conv2dWithConstraint(nProjChan1, nProjChan2, (1,1), max_norm=1, doWeightNorm=doWeightNorm, padding=0, bias=False),
            nn.BatchNorm2d(nProjChan2),
            swish()
        )

    def channelProj_op(self, x):
        y = self.channelProj(x[:,0,:,:,:])
        y = torch.unsqueeze(y, 1)
        for band in range(1, x.shape[1]):
            y = torch.cat((y, torch.unsqueeze(self.channelProj(x[:,band,:,:,:]), 1)),1)
        return y
    
    def StandTempLayer(self, m, nBands, nTimeFilter, doWeightNorm=True):
        return nn.Sequential(
            # nn.Dropout2d(p=0.5),
            # Conv2dWithConstraint(m*nBands, m*nBands*nTimeFilter, 1, max_norm=1, doWeightNorm=doWeightNorm, padding = 0),
            # nn.BatchNorm2d(m*nBands*nTimeFilter),
            # swish(),
            nn.Dropout2d(p=0.5),
            Conv2dWithConstraint(m*nBands, nTimeFilter, (1,32), stride=32, max_norm=2, doWeightNorm=doWeightNorm, bias=False),
            nn.BatchNorm2d(nTimeFilter),
            swish(),
            nn.MaxPool2d((1,4), stride=4, padding=0)
        )

    def DilateTempLayer(self, m, nBands, nTimeFilter, doWeightNorm=True):
        return nn.Sequential(
            nn.Dropout2d(p=0.5),
            Conv2dWithConstraint(m*nBands, m*nBands*nTimeFilter, 1, max_norm=2, doWeightNorm=doWeightNorm, padding = 0),
            nn.BatchNorm2d(m*nBands*nTimeFilter),
            swish(),
            nn.Dropout2d(p=0.5),
            Conv2dWithConstraint(m*nBands*nTimeFilter, m*nBands*nTimeFilter, (1,11), max_norm=1, doWeightNorm=doWeightNorm, dilation=2, bias=False, padding = (0,10)),
            nn.BatchNorm2d(m*nBands*nTimeFilter),
            swish()
        )

    def ConcatPooling(self, m, nBands, nTimeFilter):
        return nn.Sequential(
            nn.BatchNorm2d(2*m*nBands*nTimeFilter + m*nBands),
            nn.MaxPool2d((1,32), stride=32, padding=0)
        )

    def Classifier(self, m, nBands, nTimeFilter, doWeightNorm=True):
        return nn.Sequential(
            nn.Dropout(p=0.5),
            Conv2dWithConstraint(nTimeFilter+m*nBands, nTimeFilter+m*nBands, (1,1), max_norm=1, doWeightNorm=doWeightNorm, bias=False),
            nn.BatchNorm2d(nTimeFilter+m*nBands),
            # nn.MaxPool2d((1,3), stride=3, padding=0)
        )   
    
    def PointWise(self, nChan_in, nChan_out, doWeightNorm=True):
        return nn.Sequential(
            Conv2dWithConstraint(nChan_in, nChan_out, (1, 1), max_norm=2, doWeightNorm = doWeightNorm, padding = 0),
            nn.BatchNorm2d(nChan_out),
            swish()
        )
    

    def __init__(self, nChan, nProjChan=30, nBands=9, m=32, nTimeFilter=28, doWeightNorm=True, strideFactor=5, nClass=2, *args, **kwargs):
        super(MIXED_FBCNet, self).__init__()

        # channel Projection
        self.channelProj = self.ChannelProj(nChan, nProjChan, doWeightNorm=doWeightNorm)
        # shape transformation
        self.shapeTrans = self.ShapeTrans(nProjChan, nProjChan, doWeightNorm=doWeightNorm)
        # create all the parallel SCB
        self.spatialConv = self.SCB(m, nChan, nBands, doWeightNorm=doWeightNorm)
        # pointWise conv after SCB
        self.pointWise = self.PointWise(m*nBands, m*nBands)
        # Variance Layer
        self.strideFactor = strideFactor
        self.varLayer = LogVarLayer(dim=3)
        # temporal conv
        self.standTemporalLayer = self.StandTempLayer(m, nBands, nTimeFilter, doWeightNorm=doWeightNorm)
        # self.dilateTemporalLayer = self.DilateTempLayer(m, nBands, nTimeFilter, doWeightNorm=doWeightNorm)
        # Concat
        # self.concatPooling = self.ConcatPooling(m, nBands, nTimeFilter)
        # classification
        self.classifier = self.Classifier(m, nBands, nTimeFilter, doWeightNorm=doWeightNorm)
        self.lastLayer = self.LastBlock((nTimeFilter+m*nBands)*strideFactor, nClass, doWeightNorm=doWeightNorm)

    def forward(self, x):
        x = x.permute((0,3,1,2))    # n*9*22*1125
        # # Channel Projection
        # x = torch.unsqueeze(x, 3)
        # x = self.channelProj_op(x)
        # # Shape Transformation
        # for band in range(x.shape[1]):
        #     x[:,band,:,:,:] = self.shapeTrans(x[:,band,:,:,:])
        # x = torch.squeeze(x)        # n*9*35*1125   
        # Spatial Conv
        x = self.spatialConv(x)     # n*(32*9)*1*1125
        # Variance Conv
        # print(x.shape)
        x_var = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3]/self.strideFactor)]) # n*(32*9*4)*4*(1116/2)
        x_var = self.varLayer(x_var)
        x_var = torch.transpose(x_var, 2, 3)
        # print(x_var.shape)
        # Stand Temp
        x = self.standTemporalLayer(x)
        # print(x.shape)
        # # Dilate Temp
        # x_dilate = self.dilateTemporalLayer(x)
        # Concat
        # x = torch.cat((x,x_stand),dim=1)
        # x = torch.cat((x,x_dilate),dim=1)
        # x = self.concatPooling(x)
        x = torch.cat((x,x_var),dim=1)
        # classification
        x = self.classifier(x)
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        x = self.lastLayer(x)
        return x


        