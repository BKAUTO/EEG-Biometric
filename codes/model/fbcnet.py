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
        return torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))

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

class FBCNet(nn.Module):
    '''
    The data input is in a form of batch x 1 x chan x time x filterBand
    '''
    def SCB(self, m, nChan, nBands, nTimeFilter, doWeightNorm=True, *args, **kargs):
        '''
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        ''' 
        return nn.Sequential(
                Conv2dWithConstraint(nBands*nTimeFilter, m*nBands*nTimeFilter, (nChan, 1), groups= nBands*nTimeFilter,
                                     max_norm = 1 , doWeightNorm = doWeightNorm, padding = 0),
                nn.BatchNorm2d(m*nBands*nTimeFilter),
                swish()
                )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                LinearWithConstraint(inF, outF, max_norm = 0.5, doWeightNorm = doWeightNorm, *args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def ChannelProj(self, nChan, nProjChan, doWeightNorm=True):
        return nn.Sequential(
            Conv2dWithConstraint(nChan, nProjChan, (1,1), max_norm=1, doWeightNorm=doWeightNorm, padding=0),
            nn.BatchNorm2d(nProjChan),
            swish()
        )
    
    def ShapeTrans(self, nProjChan1, nProjChan2, doWeightNorm=True):
        return nn.Sequential(
            Conv2dWithConstraint(nProjChan1, nProjChan2, (1,1), max_norm=1, doWeightNorm=doWeightNorm, padding=0),
            nn.BatchNorm2d(nProjChan2),
            swish()
        )

    def channelProj_op(self, x):
        y = self.channelProj(x[:,0,:,:,:])
        y = torch.unsqueeze(y, 1)
        for band in range(1, x.shape[1]):
            y = torch.cat((y, torch.unsqueeze(self.channelProj(x[:,band,:,:,:]), 1)),1)
        return y
    
    def TempLayer(self, nBands, nTimeFilter, doWeightNorm=True):
        return nn.Sequential(
            nn.Dropout2d(p=0.5),
            Conv2dWithConstraint(nBands, nBands*nTimeFilter, (1, 9), max_norm=1, doWeightNorm = doWeightNorm, padding = 0),
            nn.BatchNorm2d(nBands*nTimeFilter),
            swish()
         )
    
    def PointWise(self, nChan_in, nChan_out, doWeightNorm=True):
        return nn.Sequential(
            Conv2dWithConstraint(nChan_in, nChan_out, (1, 1), max_norm = 2, doWeightNorm = doWeightNorm, padding = 0),
            nn.BatchNorm2d(nChan_out),
            swish()
         )

    def __init__(self, nChan, nProjChan=35, nBands=9, m=32, nTimeFilter=12, doWeightNorm=True, strideFactor=4, nClass=2, *args, **kwargs):
        super(FBCNet, self).__init__()

        # channel Projection
        self.channelProj = self.ChannelProj(nChan, nProjChan)
        # shape transformation
        self.shapeTrans = self.ShapeTrans(nProjChan, nProjChan)
        # temporal conv
        self.temporalLayer = self.TempLayer(nBands, nTimeFilter)
        # create all the parallel SCB
        self.spatialConv = self.SCB(m, nProjChan, nBands, nTimeFilter, doWeightNorm=doWeightNorm)
        # pointWise conv after SCB
        self.pointWise = self.PointWise(m*nBands, m*nBands)
        self.strideFactor = strideFactor
        self.varLayer = LogVarLayer(dim=3)
        self.lastLayer = self.LastBlock(m*nBands*strideFactor*nTimeFilter, nClass, doWeightNorm=doWeightNorm)

        # # weight initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        x = x.permute((0,3,1,2))    # n*9*22*1124
        # Channel Projection
        x = torch.unsqueeze(x, 3)
        x = self.channelProj_op(x)
        # Shape Transformation
        for band in range(x.shape[1]):
            x[:,band,:,:,:] = self.shapeTrans(x[:,band,:,:,:])
        x = torch.squeeze(x)        # n*9*35*1124   
        # Temporal Conv
        x = self.temporalLayer(x)   # n*(9*4)*35*1116 
        x = self.spatialConv(x)     # n*(32*9*4)*1*1116
        # x = self.pointWise(x)    # n*(32*9*4)*1*1116
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3]/self.strideFactor)]) # n*(32*9*4)*4*(1116/2)
        x = self.varLayer(x)
        print(x.shape)
        x = torch.flatten(x, start_dim=1)
        x = self.lastLayer(x)
        return x


        