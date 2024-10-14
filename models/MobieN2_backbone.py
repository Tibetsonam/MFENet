import torch
import torch.nn as nn
import torch.nn.functional as F
# from  MobieNet.MobieNet_v2 import mobilenet_v2
from models.MobieNet.MobieNet_v2 import mobilenet_v2
from timm.models.layers import DropPath
from models.DepthMobie import depth_mobie
from models.FlowMobie import flow_mobie
from models.SC_branch import SC_mobie
import inspect


from try_idea.oct import NormalOctaveConv
from try_idea.edgeenhance import EdgeEnhancedModule
from try_idea.CA import CoordAtt

class out_block(nn.Module):
    def __init__(self, infilter):
        super(out_block, self).__init__()
        self.conv1 = nn.Sequential(
            *[nn.Conv2d(infilter, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)])
        self.conv2 = nn.Conv2d(64, 1, 1)
        

    def forward(self, x, H, W):
        x = F.interpolate(self.conv1(x), (H, W), mode='bilinear', align_corners=True)
        y = self.conv2(x)
        # print(y)
        return y


class decoder_stage(nn.Module):
    def __init__(self, infilter, midfilter, outfilter):
        super(decoder_stage, self).__init__()
        self.layer = nn.Sequential(
            *[DepthwiseSeparableConv2d(infilter, midfilter, 3, padding=1), 
              nn.BatchNorm2d(midfilter),
              nn.ReLU(inplace=True),
              DepthwiseSeparableConv2d(midfilter, midfilter, 3, padding=1), 
              nn.BatchNorm2d(midfilter),
              nn.ReLU(inplace=True),
              DepthwiseSeparableConv2d(midfilter, outfilter, 3, padding=1), 
              nn.BatchNorm2d(outfilter),
              nn.ReLU(inplace=True)])
      
    def forward(self, x):
        return self.layer(x)


# class BasicConv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,dilation=1):
#         # padding = (kernel_size - 1) // 2
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
#                               padding=padding,dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return self.relu(x)
    
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_planes, dilation=dilation, bias=False)
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

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
class BasicRFB_a(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_a, self).__init__()
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
# StarNet layer 
class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


def upsample(x, size):
    return F.interpolate(x, size, mode='bilinear', align_corners=True)
class change_channel(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(change_channel, self).__init__()
        self.conv1 =  nn.Conv2d(in_channel, out_channel, kernel_size=1)
    def forward(self, x):
        result = self.conv1(x)
        return result
class Model(nn.Module):
    def __init__(self, mode):   #test
    # def __init__(self, mode,model_path): #trian
        super(Model, self).__init__()
        self.rgb_bkbone = mobilenet_v2(3, pretrained=True)
        self.depth_bkbone = SC_mobie(pretrained=False)
        # self.depth_bkbone = depth_mobie(pretrained=False)

        # self.squeeze5 = nn.Sequential(nn.Conv2d(320, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # self.squeeze4 = nn.Sequential(nn.Conv2d(96, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d(32, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(24, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze1 = nn.Sequential(nn.Conv2d(16, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))


        # self.squeeze4_depth = nn.Sequential(nn.Conv2d(96, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3_depth = nn.Sequential(nn.Conv2d(32, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2_depth = nn.Sequential(nn.Conv2d(24, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze1_depth = nn.Sequential(nn.Conv2d(16, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.catconv1 = DepthwiseSeparableConv2d(128, 64, 3, stride=1, padding=1)
        self.catconv2 = DepthwiseSeparableConv2d(128, 64, 3, stride=1, padding=1)
        self.catconv3 = DepthwiseSeparableConv2d(128, 32, 3, stride=1, padding=1)
        # self.catconv4 = DepthwiseSeparableConv2d(128, 64, 3, stride=1, padding=1)
        
        # Fdem mudule
        self.Oct1 = NormalOctaveConv(0,64,64,64,kernel=3,stride=1,padding=1)		#   L H L H  L2L  L2H
        self.Oct2 = NormalOctaveConv(64,64,64,64,kernel=3,stride=1,padding=1)
        self.Oct3 = NormalOctaveConv(64,64,64,64,kernel=3,stride=1,padding=1)
        self.Oct4 = NormalOctaveConv(64,96,0,96,kernel=3,stride=1,padding=1)      
        # self.Oct5 = NormalOctaveConv(0,320,0,320,kernel=3,stride=1,padding=1)
        self.edge = EdgeEnhancedModule(64,64)

        self.ca1 = CoordAtt(64,64)
        self.ca2 = CoordAtt(64,64)	
        self.ca3 = CoordAtt(64,64)	
        self.ca4 = CoordAtt(96,96)	
        self.ca5 = CoordAtt(320,320)	
        # self.RFB1 = BasicRFB_a(64,64)  #
        # self.RFB2 = BasicRFB_a(64,64)  #
        # self.RFB3 = BasicRFB_a(32,32)  #
        # self.RFB4 = BasicRFB_a(96,96)  #
        # self.RFB5 = BasicRFB_a(320,320)  #
        #octconv layer
        # self.Oct1 = OctaveConv(0,64,64,64,kernel=3,stride=1,padding=1)		#   L H L H L2L  L2H
        # self.Oct2 = OctaveConv(64,64,64,64,kernel=3,stride=1,padding=1)
        # self.Oct3 = OctaveConv(0,64,64,64,kernel=3,stride=1,padding=1)
        # self.Oct4 = OctaveConv(64,64,96,96,kernel=3,stride=1,padding=1)      
        # self.Oct5 = OctaveConv(96,96,0,320,kernel=3,stride=1,padding=1)	
        # 	
        # self.Oct3 = OctaveConv(64,64,64,64,kernel=3,stride=1,padding=1)
        # self.Oct4 = OctaveConv(64,64,96,96,kernel=3,stride=1,padding=1)      
        # self.Oct5 = OctaveConv(96,96,0,320,kernel=3,stride=1,padding=1)
        self.change_channel1 = change_channel(32,64)
        # self.change_channel2 = change_channel(96,320)
        # ------------decoder----------------#
        self.decoder5 = decoder_stage(320, 128, 64)  #
        self.decoder4 = decoder_stage(160, 128, 64)  #
        self.decoder3 = decoder_stage(128, 128, 64)  #
        self.decoder2 = decoder_stage(128, 128, 64)  #
        self.decoder1 = decoder_stage(128, 128, 64)  #
        
        self.out5 = out_block(64)
        self.out4 = out_block(64)
        self.out3 = out_block(64)
        self.out2 = out_block(64)
        self.out1 = out_block(64)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    
    def forward(self, image, flow, depth):

        rgb_out1, rgb_out2, rgb_out3= self.rgb_bkbone(image) # 8 16 224 224 #8 24 112 112 # 8 32 56 56 
        i = depth,flow
        depth_out1, depth_out2, depth_out3 = self.depth_bkbone(i) 

        rgb_out1, rgb_out2, rgb_out3= self.squeeze1(rgb_out1), \
                                                 self.squeeze2(rgb_out2), \
                                                 self.squeeze3(rgb_out3)
                                                

        depth_out1, depth_out2,depth_out3 = self.squeeze1_depth(depth_out1), \
                                                         self.squeeze2_depth(depth_out2),\
                                                         self.squeeze3_depth(depth_out3)
                                                        #  self.squeeze4_depth(depth_out4)

        fusion1 = self.catconv1(torch.cat([rgb_out1, depth_out1], dim=1)) # 8 64 224 224  
        F1 = fusion1.size()
        fusion2 = self.catconv2(torch.cat([rgb_out2, depth_out2], dim=1)) # 8 64 112 112   
        F2 = fusion2.size()
        fusion3_ = self.catconv3(torch.cat([rgb_out3, depth_out3], dim=1)) # 8 32 56 56
        fusion3 = self.change_channel1(fusion3_)
        # fusion3 = self.catconv2(torch.cat([rgb_out3, depth_out3,], dim=1)) # 8 64 56 56
        F3 = fusion3_.size()
        
        rgb_out4,rgb_out5 = self.rgb_bkbone(fusion3_)

        # rgb_out4 = self.RFB1(rgb_out4)
        r4 = rgb_out4.size()  # 8 96 28 28  
        r5 = rgb_out5.size()  # 8 320 28  28
        
        # OCT_1
        i = 0, fusion1 
        X_l,X_h = self.Oct1(i)
        X_h = self.ca1(X_h)
        fusion1 =  X_h
        #OCT_2
        X_l = nn.AdaptiveAvgPool2d((56, 56))(X_l)
        i = X_l, fusion2 
        X_l,X_h = self.Oct2(i) 
        edge_feature,f1_edge = self.edge(X_l)                  
        edge_up1 = self.out1(edge_feature, 448, 448) 
        # X_l = edge_feature
        X_h = self.ca2(X_h)
        fusion2 = fusion2 + X_h
        #OCT_3
        X_l = nn.AdaptiveAvgPool2d((28, 28))(X_l)
        i = X_l, fusion3 
        X_l,X_h = self.Oct3(i) 
        edge_feature,f1_edge = self.edge(X_l)                  
        edge_up2 = self.out1(edge_feature, 448, 448) 
        # X_l = edge_feature
        X_h = self.ca3(X_h)
        fusion3 = fusion3 + X_h
        #OCT_4
        X_l = nn.AdaptiveAvgPool2d((14, 14))(X_l)
        i = X_l, rgb_out4
        # X_lsize = X_l.size()
        # X_ = rgb_out4.size() 
        X_l,X_h = self.Oct4(i) 
        X_h = self.ca4(X_h)
        rgb_out4 = rgb_out4 + X_h
        # X_lsize = X_l.size()
        #OCT_5
        # i = X_l, rgb_out5 
        # X_l,X_h = self.Oct5(i) 
        # X_h = self.ca5(X_h)
        # rgb_out5 = rgb_out5 + X_h         
        rgb_out5 = self.ca5(rgb_out5)

        


        edge_up = []
        edge_up.append(edge_up1)
        edge_up.append(edge_up2)
        edge_up_final = edge_up1



        feature5 = self.decoder5(rgb_out5) #64
        B, C, H, W = rgb_out4.size()
        feature4 = self.decoder4(
            torch.cat((feature5, rgb_out4), 1))
        B, C, H, W = fusion3.size()
        feature3 = self.decoder3(
            torch.cat((F.interpolate(feature4, (H, W), mode='bilinear', align_corners=True), fusion3), 1))
        B, C, H, W = fusion2.size() #224 224
        feature2 = self.decoder2(
            torch.cat((F.interpolate(feature3, (H, W), mode='bilinear', align_corners=True), fusion2), 1))
        B, C, H, W = fusion1.size() #8 64 224 224 
        feature1 = self.decoder1(
            torch.cat((F.interpolate(feature2, (H, W), mode='bilinear', align_corners=True), fusion1), 1))
        


        decoder_out5 = self.out5(feature5, H * 2, W * 2)
        # print(decoder_out5)
        B1, C1, H1, W1 = feature5.size() # 64 28 28 
        B1, C1, H1, W1 = decoder_out5.size() # 1 448 448 

        decoder_out4 = self.out4(feature4, H * 2, W * 2)
        B1, C1, H1, W1 = feature4.size() # 64 28 28 
        B1, C1, H1, W1 = decoder_out4.size() # 1 448 448 

        decoder_out3 = self.out3(feature3, H * 2, W * 2)
        B1, C1, H1, W1 = feature3.size() # 64 56 56 
        B1, C1, H1, W1 = decoder_out3.size() #

        decoder_out2 = self.out2(feature2, H * 2, W * 2)
        B1, C1, H1, W1 = feature2.size() # 64 112 112 
        B1, C1, H1, W1 = decoder_out2.size() #
        
        decoder_out1 = self.out1(feature1, H * 2, W * 2)
        B1, C1, H1, W1 = feature1.size() # 64 224 224 
        B1, C1, H1, W1 = decoder_out1.size() #

        return decoder_out1, decoder_out2, decoder_out3, decoder_out4,decoder_out5,edge_up_final
