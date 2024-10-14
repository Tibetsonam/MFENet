import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalOctaveConv(nn.Module):
    def __init__(self,Lin_channel,Hin_channel,Lout_channel,Hout_channel,
            kernel, stride=1, padding=0, dilation=1, alpha=0.75, groups=1, up_kwargs={'mode': 'nearest'}):
        super(NormalOctaveConv, self).__init__()
        self.stride = stride
        self.Lout_channel  = Lout_channel
        self.Hout_channel  = Hout_channel

        if Lout_channel != 0 and Lin_channel != 0:
            self.convL2L = nn.Conv2d(Lin_channel,Lout_channel, kernel,stride,padding)
            self.convH2L = nn.Conv2d(Hin_channel,Lout_channel, kernel,stride,padding)
            self.convL2H = nn.Conv2d(Lin_channel,Hout_channel, kernel,stride,padding)
            self.convH2H = nn.Conv2d(Hin_channel,Hout_channel, kernel,stride,padding)
        elif Lout_channel == 0 and Lin_channel != 0:
            self.convL2L = None
            self.convH2L = None
            self.convL2H = nn.Conv2d(Lin_channel,Hout_channel, kernel,stride,padding)
            self.convH2H = nn.Conv2d(Hin_channel,Hout_channel, kernel,stride,padding)
        elif Lout_channel != 0 and Lin_channel == 0:
            self.convL2L = None
            self.convH2L = nn.Conv2d(Hin_channel,Lout_channel, kernel,stride,padding)
            self.convL2H = None
            self.convH2H = nn.Conv2d(Hin_channel,Hout_channel, kernel,stride,padding)
        else:
            self.convL2L = None
            self.convH2L = None
            self.convL2H = None
            self.convH2H = nn.Conv2d(Hin_channel,Hout_channel, kernel,stride,padding)
        self.upsample = nn.Upsample(scale_factor=2)
        self.pool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=False)

        self.bn_h =  nn.BatchNorm2d(Hout_channel)
        self.bn_l =  nn.BatchNorm2d(Lout_channel)

        # self.bn  = nn.BatchNorm2d(Hout_channel)

    def forward(self,x):
        Lx,Hx = x
        if self.convL2L is not None:
            L2Ly = self.convL2L(Lx)
        else:
            L2Ly = 0
        if self.convL2H is not None:
            L2Hy = self.upsample(self.convL2H(Lx))
        else:
            L2Hy = 0
        if self.convH2L is not None:
            H2Ly = self.convH2L(self.pool(Hx))
        else:
            H2Ly = 0
        if self.convH2H is not None:
            H2Hy = self.convH2H(Hx)
        else:
            H2Hy = 0

        Lx = L2Ly+H2Ly
        Hx = L2Hy+H2Hy

        #xinjiade 
        if self.Lout_channel != 0: 
            Lx = self.relu(self.bn_l(Lx))
        if self.Hout_channel != 0: 
            Hx = self.relu(self.bn_h(Hx))
        # if torch.is_tensor(Lx):
        #     # Lx = self.bn(Lx)           
        #     Lx = self.relu(Lx)
        # if torch.is_tensor(Hx):
        #     # Hx = self.bn(Hx)   
        #     Hx = self.relu(Hx)
        return Lx,Hx