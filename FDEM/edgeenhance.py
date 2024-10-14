
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class EdgeEnhancedModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeEnhancedModule, self).__init__()
        self.numinchannel  = in_channels
        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        # Initialize Sobel kernels
        sobel_kernel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        # Repeat the kernels for each input channel
        sobel_kernel_x = sobel_kernel_x.repeat(in_channels, 1, 1, 1)
        sobel_kernel_y = sobel_kernel_y.repeat(in_channels, 1, 1, 1)

        self.sobel_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)

        self.conv3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.combined_conv = nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.change_conv = nn.Conv2d(in_channels*2, out_channels, kernel_size=1, padding=0)
        self.DWconv = _DSConv(in_channels,in_channels)
        
    def forward(self,f1_d):
        # Apply Sobel filters to the input features
        # f1_d_x = F.relu(self.sobel_x(f1_d))
        # f1_d_y = F.relu(self.sobel_y(f1_d))
        f1_d = self.DWconv(f1_d)
        I_x = self.sobel_x(f1_d)
        I_y = self.sobel_y(f1_d)
        # 计算边缘强度
        edges = torch.sqrt(I_x ** 2 + I_y ** 2)
        # # 对边缘强度进行σ操作（非线性激活函数，如sigmoid）
        f1_edge = torch.sigmoid(edges)
        # f2_d_x = F.relu(self.sobel_x(f2_d))
        # f2_d_y = F.relu(self.sobel_y(f2_d))
        
        # Combine Sobel x and y results
        # f1_edge =  torch.sigmoid(self.change_conv(torch.cat((I_x, I_y), dim=1)))
       
        # f2_edge = f2_d_x + f2_d_y
        
        # Multiply edges with features

        f1_enhanced = f1_d + f1_edge*0.7
        # f2_enhanced = f2_d * f2_edge
         
        f_e = self.final_conv(f1_enhanced)
        return  f_e,f1_edge
    


    