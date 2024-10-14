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

def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

class Shift8(nn.Module):
    def __init__(self, groups=4, stride=1, mode='constant') -> None:
        super().__init__()
        self.g = groups
        self.mode = mode
        self.stride = stride

    def forward(self, x):
        b, c, h, w = x.shape
        out = torch.zeros_like(x)

        pad_x = F.pad(x, pad=[self.stride for _ in range(4)], mode=self.mode)
        assert c == self.g * 8

        cx, cy = self.stride, self.stride
        stride = self.stride
        out[:,0*self.g:1*self.g, :, :] = pad_x[:, 0*self.g:1*self.g, cx-stride:cx-stride+h, cy:cy+w]
        out[:,1*self.g:2*self.g, :, :] = pad_x[:, 1*self.g:2*self.g, cx+stride:cx+stride+h, cy:cy+w]
        out[:,2*self.g:3*self.g, :, :] = pad_x[:, 2*self.g:3*self.g, cx:cx+h, cy-stride:cy-stride+w]
        out[:,3*self.g:4*self.g, :, :] = pad_x[:, 3*self.g:4*self.g, cx:cx+h, cy+stride:cy+stride+w]

        out[:,4*self.g:5*self.g, :, :] = pad_x[:, 4*self.g:5*self.g, cx+stride:cx+stride+h, cy+stride:cy+stride+w]
        out[:,5*self.g:6*self.g, :, :] = pad_x[:, 5*self.g:6*self.g, cx+stride:cx+stride+h, cy-stride:cy-stride+w]
        out[:,6*self.g:7*self.g, :, :] = pad_x[:, 6*self.g:7*self.g, cx-stride:cx-stride+h, cy+stride:cy+stride+w]
        out[:,7*self.g:8*self.g, :, :] = pad_x[:, 7*self.g:8*self.g, cx-stride:cx-stride+h, cy-stride:cy-stride+w]

        #out[:, 8*self.g:, :, :] = pad_x[:, 8*self.g:, cx:cx+h, cy:cy+w]
        return out
# SC resdidual block
class ResidualBlockShift(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-Shift-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64,inchannel = 64, out_channels= 32,stride=2,res_scale=1,pytorch_init=False):
        super(ResidualBlockShift, self).__init__()
        self.res_scale = res_scale
        self.use_shortcut = stride == 1 and inchannel == out_channels
        self.conv1 = nn.Conv2d(inchannel, num_feat, kernel_size=1,stride=stride)  # 尺寸减小一半 channel = 64
        self.shift = Shift8(groups=num_feat//8, stride=1)
        self.conv2 = nn.Conv2d(num_feat, out_channels, kernel_size=1)            # channel = out_channels,channel 改为 期望输入 16 24 32
        self.relu6 = nn.ReLU6(inplace=True) 
        self.bn = nn.BatchNorm2d(out_channels)  
        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2,self.bn], 0.1)
        
    def forward(self, x):  
        identity = x # 64 448 448       16 224 224 
        out = self.shift(self.conv1(x))     # 64 224 224  64 224 224 
        out = self.conv2(self.relu6(out))   # 16 224 224  16 224 224 
        out = self.bn(out) # 224 
        out = out * self.res_scale
        if self.use_shortcut:
            out = identity + out * self.res_scale
        return out
    
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

class BasicRFB_a(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_a, self).__init__()
        assert in_planes == out_planes, "in_planes and out_planes must be the same"
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
            BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        out = out * self.scale + x  # Use x directly as shortcut
        out = self.relu(out)

        return out
    
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class DepthBranch(nn.Module):
    # def __init__(self, c1=8, c2=16, c3=32, c4=48, c5=320, **kwargs):
    # def __init__(self, pretrained, model_path, **kwargs): 
    def __init__(self, pretrained,num_in_ch=128, num_out_ch=56, num_feat=64):
    
        super(DepthBranch, self).__init__()

        # self.conv_last = nn.Conv2d(num_feat, num_out_ch, kernel_size=1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # if self.upscale == 4:
        #     default_init_weights(self.upconv2, 0.1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat,1)
                # initialization
        default_init_weights([self.conv_first], 0.1)
        self.bottleneck1 = _make_layer(ResidualBlockShift,num_block=4,num_feat=num_feat,inchannel=num_feat,outchannel=16,stride=2)  # c = 16 224 224

        self.bottleneck2 = _make_layer(ResidualBlockShift,num_block=7,num_feat=num_feat,inchannel=16,outchannel=24,stride=2)
        self.bottleneck3 = _make_layer(ResidualBlockShift,num_block=4,num_feat=num_feat,inchannel=24,outchannel=32,stride=2)
        # self.DWconv = _DSConv(3,64)
        self.LBockconv = LinearBottleneck(3,64,stride=1)
        
        self.RFB1 = BasicRFB_a(64,64)  #
        # if pretrained:
        #         self._load_pretrained_model(model_path)
    def forward(self, x):
        depth,flow = x
        #channel 变为64 
        depth = self.LBockconv(depth) 
        flow = self.LBockconv(flow)  #74.3 9.8mb
        depth = self.RFB1(depth)    
        flow = self.RFB1(flow)      #74.9 10mb  74.6 
        #batch_size 8

        batch_size, channels, height, width = depth.shape
        # 创建一个空的张量用于存放交替排列后的结果
        concatenated_input = torch.empty(batch_size, channels * 2, height, width, device=depth.device)
        # 交替排列 depth 和 flow 的通道
        concatenated_input[:, 0::2, :, :] = depth
        concatenated_input[:, 1::2, :, :] = flow

        crossfusion = concatenated_input.size() # 128 
        feat = self.lrelu(self.conv_first(concatenated_input))

        # size = x.size()[2:]  # 448 448 
        # size = x.size()
        # feat = []
        
        # feat = self.lrelu(self.conv_first(x))              

        x1 = self.bottleneck1(feat)
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

        # feat.append(x1)
        # feat.append(x2)
        # feat.append(x3)
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

def _make_layer( block, num_block, num_feat,inchannel, outchannel,stride=1):
    layers = []
    layers.append(block(num_feat, inchannel,outchannel, stride))
    for i in range(1, num_block):
        layers.append(block(num_feat,outchannel, outchannel, 1))
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

def SC_mobie(pretrained):
    model = DepthBranch(pretrained,num_in_ch=128, num_out_ch=56, num_feat=64)
    return model