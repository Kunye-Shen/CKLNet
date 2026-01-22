import torch.nn as nn
import torch.nn.functional as F
from .basic import *

class CKLNet_Encoder(nn.Module):
    def __init__(self, num_classes):
        super(CKLNet_Encoder, self).__init__()
        #MI-based real-time backbone
        ##stage 1
        self.layer1 = convbnrelu(3, 16, k=3, s=2, p=1)

        ##stage 2
        self.layer2 = nn.Sequential(
                DSConv3x3(16, 32, stride=2),
                MI_Module(32),
                MI_Module(32),
                MI_Module(32)
                )

        ##stage 3
        self.layer3 = nn.Sequential(
                DSConv3x3(32, 64, stride=2),
                MI_Module(64),
                MI_Module(64),
                MI_Module(64),
                MI_Module(64)
                )
                
        ##stage 4
        self.layer4 = nn.Sequential(
                DSConv3x3(64, 96, stride=2),
                MI_Module(96),
                MI_Module(96),
                MI_Module(96),
                MI_Module(96),
                MI_Module(96),
                MI_Module(96)
                )

        ##stage 5
        self.layer5 = nn.Sequential(
                DSConv3x3(96, 128, stride=2),
                MI_Module(128),
                MI_Module(128),
                MI_Module(128)
                )
        
        ##class-information encoding layer
        self.layer6 = nn.Sequential(
                DSConv3x3(96, 128, stride=2),
                MI_Module(128),
                MI_Module(128),
                MI_Module(128)
                )
        
        ##classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        #MI-based real-time backbone
        ##stage 1
        score1 = self.layer1(x)
        ##stage 2
        score2 = self.layer2(score1)
        ##stage 3
        score3 = self.layer3(score2)
        ##stage 4
        score4 = self.layer4(score3)
        ##stage 5
        score5 = self.layer5(score4)
        
        ##class-information encoding layer
        score6 = self.layer6(score4)

        ##classification head
        Cls = self.avgpool(score6)
        Cls = Cls.view(Cls.size(0), -1)
        Cls = self.fc(Cls)
        Cls = torch.argmax(Cls, dim=1)

        return score1, score2, score3, score4, score5, Cls

class CKLNet_Decoder(nn.Module):
    def __init__(self):
        super(CKLNet_Decoder, self).__init__()
        ##stage 5
        self.decoder5 = nn.Sequential(
                DSConv3x3(128,128, dilation=2),
                DSConv3x3(128,96, dilation=1)
        )

        ##stage 4
        self.decoder4 = nn.Sequential(
                DSConv3x3(96,96, dilation=2),
                DSConv3x3(96,64, dilation=1)
        )

        ##stage 3
        self.decoder3 = nn.Sequential(
                DSConv3x3(64,64, dilation=2),
                DSConv3x3(64,32, dilation=1)
        )

        ##stage 2
        self.decoder2 = nn.Sequential(
                DSConv3x3(32,32, dilation=2),
                DSConv3x3(32,16, dilation=1)
        )

        ##stage 1
        self.decoder1 = nn.Sequential(
                DSConv3x3(16,16, dilation=2),
                DSConv3x3(16,16, dilation=1)
        )

        #Output
        self.conv_out1 = ConvOut(in_channel=16)

    def forward(self, score1, score2, score3, score4, score5, output_shape=(200,200)):
        ##stage 5
        scored5 = self.decoder5(score5)
        t = interpolate(scored5, score4.size()[2:])

        ##stage 4
        scored4 = self.decoder4(score4 + t)
        t = interpolate(scored4, score3.size()[2:])

        ##stage 3
        scored3 = self.decoder3(score3 + t)
        t = interpolate(scored3, score2.size()[2:])

        ##stage 2
        scored2 = self.decoder2(score2 + t)
        t = interpolate(scored2, score1.size()[2:])

        ##stage 1
        scored1 = self.decoder1(score1 + t)

        #Output
        out = self.conv_out1(scored1)

        out = interpolate(out, output_shape)

        return out

interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)