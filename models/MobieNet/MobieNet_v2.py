from torch import nn
from torchvision.models import MobileNetV2
import torch


class MobileNetV2Encoder(MobileNetV2):
    """
    MobileNetV2Encoder inherits from torchvision's official MobileNetV2. It is modified to
    use dilation on the last block to maintain output stride 16, and deleted the
    classifier block that was originally used for classification. The forward method
    additionally returns the feature maps at all resolutions for decoder's use.
    """

    def __init__(self, in_channels,pretrained,model_path,norm_layer=None):
        super().__init__()

        # Replace first conv layer if in_channels doesn't match.
        if in_channels != 3:
            self.features[0][0] = nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False)

        # Remove last block
        self.features = self.features[:-1]

        # Change to use dilation to maintain output stride = 16
        self.features[14].conv[1][0].stride = (1, 1)
        for feature in self.features[15:]:
            feature.conv[1][0].dilation = (2, 2)
            feature.conv[1][0].padding = (2, 2)

        # Delete classifier
        del self.classifier

        self.layer1 = nn.Sequential(self.features[0], self.features[1])
        self.layer2 = nn.Sequential(self.features[2], self.features[3])
        self.layer3 = nn.Sequential(self.features[4], self.features[5], self.features[6])
        self.layer4 = nn.Sequential(self.features[7], self.features[8], self.features[9], self.features[10],
                                    self.features[11], self.features[12], self.features[13])
        self.layer5 = nn.Sequential(self.features[14], self.features[15], self.features[16], self.features[17])
        if pretrained:
            self._load_pretrained_model(model_path)
    def forward(self, x):
        if x.size()[2:] == (56,56):
            x = self.features[7](x)
            x = self.features[8](x)
            x = self.features[9](x)
            x = self.features[10](x)
            x = self.features[11](x)
            x = self.features[12](x)
            x = self.features[13](x)
            x4 = x # 1/16
            B,C,H,W = x4.size()  # 8 96 28 28
            x = self.features[14](x)
            x = self.features[15](x)
            x = self.features[16](x)
            x = self.features[17](x)
            x5 = x # 1/16
            B,C,H,W = x5.size()  # 8 320 28 28   
            return x4,x5
        else:
            x0 = x  # 1/1
            x = self.features[0](x)
            x = self.features[1](x)
            x = x 
            x1 = x  # 1/2
            B,C,H,W = x1.size()  # 8 16 224 224 
            x = self.features[2](x)
            x = self.features[3](x)
            x2 = x  # 1/4
            B,C,H,W = x2.size()  # 8 24 112 112  
            x = self.features[4](x)
            x = self.features[5](x)
            x = self.features[6](x)
            x3 = x  # 1/8
            B,C,H,W = x3.size()  # 8 32 56 56  
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

def mobilenet_v2(input,pretrained):
    model = MobileNetV2Encoder(input,pretrained,model_path="./models/MobieNet/pre_train/mobilenet_v2-b0353104.pth")
    return model