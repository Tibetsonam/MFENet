import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import time
# import timm
# from mobilenet import MobileNetV2Encoder


import os
import torch
import torch.nn as nn
import torch.nn.functional as F



# def upsample(x, size):
#     return F.interpolate(x, size, mode='bilinear', align_corners=True)

# def initialize_weights(model):
#     m = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
#     pretrained_dict = m.state_dict()
#     all_params = {}
#     for k, v in model.state_dict().items():
#         if k in pretrained_dict.keys() and v.shape == pretrained_dict[k]:
#             v = pretrained_dict[k]
#             all_params[k] = v
#     # assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
    # model.load_state_dict(all_params,strict=False)

class DepthBranch(nn.Module):
    # def __init__(self, c1=8, c2=16, c3=32, c4=48, c5=320, **kwargs):
    def __init__(self, pretrained,num_in_ch=6, num_out_ch=56, num_feat=64, **kwargs):
        super(DepthBranch, self).__init__()
        self.bottleneck1 = _make_layer(LinearBottleneck, 6, 16, blocks=4, t=3, stride=2)
        self.bottleneck2 = _make_layer(LinearBottleneck, 16, 24, blocks=7, t=3, stride=2)
        self.bottleneck3 = _make_layer(LinearBottleneck, 24, 32, blocks=4, t=3, stride=2)
        # self.bottleneck4 = _make_layer(LinearBottleneck, 32, 96, blocks=3, t=2, stride=2)
        # self.bottleneck5 = _make_layer(LinearBottleneck, 96, 320, blocks=1, t=2, stride=1)

        # self.conv_s_d = _ConvBNReLU(320,1,1,1)

        # nn.Sequential(_DSConv(c3, c3 // 4),
        #                           nn.Conv2d(c3 // 4, 1, 1), )
        # if pretrained:
        #         self._load_pretrained_model(model_path)
    def forward(self, x):
        # size = x.size()[2:]  # 448 448 
        # size = x.size()
        depth,flow = x
        feat = []
        batch_size, channels, height, width = depth.shape
        # 创建一个空的张量用于存放交替排列后的结果
        concatenated_input = torch.empty(batch_size, channels * 2, height, width, device=depth.device)
        # 交替排列 depth 和 flow 的通道
        concatenated_input[:, 0::2, :, :] = depth
        concatenated_input[:, 1::2, :, :] = flow

        x1 = self.bottleneck1(concatenated_input)
        B,C,H,W = x1.size()      # 8 16 224 224
        x2 = self.bottleneck2(x1)
        B,C,H,W = x2.size()      # 8 24 112 112 
        x3 = self.bottleneck3(x2)
        B,C,H,W = x3.size()      # 8 32 56 56 
        # x4 = self.bottleneck4(x3)
        # B,C,H,W = x4.size()      # 4 96 28 28 
        # x5 = self.bottleneck5(x4)
        # B,C,H,W = x5.size()      # 4 320 28 28 
        # s_d = self.conv_s_d(x5)

        feat.append(x1)
        feat.append(x2)
        feat.append(x3)
        # feat.append(x4)
        # feat.append(x5)
        return x1,x2,x3
    
    def _load_pretrained_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict: 
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict,strict=False)

class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv(x)


# class _DSConv(nn.Module):
#     """Depthwise Separable Convolutions"""

#     def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
#         super(_DSConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
#             nn.BatchNorm2d(dw_channels),
#             nn.ReLU(True),
#             nn.Conv2d(dw_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(True)
#         )

#     def forward(self, x):
#         return self.conv(x)
# self.bottleneck1 = _make_layer(LinearBottleneck, 1, 16, blocks=1, t=3, stride=2)
def _make_layer( block, inplanes, planes, blocks, t=6, stride=1):
    layers = []
    layers.append(block(inplanes, planes, t, stride))
    for i in range(1, blocks):
        layers.append(block(planes, planes, t, 1))
    return nn.Sequential(*layers)

class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)



    """LinearBottleneck used in MobileNetV2"""
class LinearBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


# class PyramidPooling(nn.Module):
#     """Pyramid pooling module"""

#     def __init__(self, in_channels, out_channels, **kwargs):
#         super(PyramidPooling, self).__init__()
#         inter_channels = int(in_channels / 4)
#         self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
#         self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
#         self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
#         self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
#         self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

#     def pool(self, x, size):
#         avgpool = nn.AdaptiveAvgPool2d(size)
#         return avgpool(x)

#     def forward(self, x):
#         size = x.size()[2:]
#         feat1 = upsample(self.conv1(self.pool(x, 1)), size)
#         feat2 = upsample(self.conv2(self.pool(x, 2)), size)
#         feat3 = upsample(self.conv3(self.pool(x, 3)), size)
#         feat4 = upsample(self.conv4(self.pool(x, 6)), size)
#         x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
#         x = self.out(x)
#         return x




# class BasicConv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, activation='relu'):
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_planes, out_planes,
#                               kernel_size=kernel_size, stride=stride,
#                               padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.activation = activation
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return  self.relu(x) if self.activation=='relu' \
#         else self.sigmoid(x) if self.activation=='sigmoid' \
#         else x

def depth_mobie(pretrained):
    model = DepthBranch(pretrained,num_in_ch=6, num_out_ch=56, num_feat=64)
    # model = DepthBranch(pretrained,model_path="./models/MobieNet/pre_train/mobilenet_v2-b0353104.pth")
    return model