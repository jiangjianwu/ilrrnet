"""Pyramid Scene Parsing Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .segbase import SegBaseModel
from .fcn import _FCNHead

__all__ = ["ICRNet_V2PSP"]

class ICRNet_V2PSP(SegBaseModel):
    def __init__(self, nclass, backbone='resnet50', aux=False, pretrained=True, norm_layer=nn.BatchNorm2d, **kwargs):
        # Pretrained layers | Backbone = Resnet 50
        super(ICRNet_V2PSP, self).__init__(sum(nclass), aux, backbone, pretrained_base=pretrained, **kwargs)
        
        # Common feature
        self.head = _ICRHead(nclass, norm_layer=norm_layer, **kwargs)
        if self.aux:
            self.auxlayer = _FCNHead(1024, nclass, norm_layer=norm_layer, **kwargs)

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

        # Feature extraction based on classes
        self.psp_semseg = _PyramidPooling(512, norm_layer=norm_layer, norm_kwargs = None)
        self.psp_rational = _PyramidPooling(512, norm_layer=norm_layer, norm_kwargs = None)
        # Multi task output
        self.out_semseg = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(512, nclass[4], 1)
        )
        self.out_rational = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(512, 256, 1, padding=1, bias=True),
            nn.Conv2d(256, nclass[5], 1)
        )
     
    def forward(self, x):
        size = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c4)

        x_semseg = self.psp_semseg(x)
        x_semseg = self.out_semseg(x_semseg)
        outputs.append(F.interpolate(x_semseg, size, mode='bilinear', align_corners=True))
        x_rational = self.psp_rational(x)
        x_rational = self.out_rational(x_rational)
        outputs.append(F.interpolate(x_rational, size, mode='bilinear', align_corners=True))

        return tuple(outputs)

class _ICRHead(nn.Module):
    def __init__(self, nclass = [18, 2], norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_ICRHead, self).__init__()
        # ASPP
        self.aspp = _ASPP(2048, [12, 24, 36], norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
        self.block_aspp = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1)
        )
        
        # PSP
        self.psp = _PyramidPooling(2048, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.block_psp = nn.Sequential(
            nn.Conv2d(4096, 512, 3, padding=1, bias=False),
            norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x1 = self.aspp(x)
        x1 = self.block_aspp(x1) # ASPP : 512
        x2 = self.psp(x)
        x2 = self.block_psp(x2) # PSP : 512
        return x1 + x2


# ============= PSP =============
def _ICR1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
        nn.ReLU(True)
    )

class _PyramidPooling(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(_PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)
        self.avgpool3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = _ICR1x1Conv(in_channels, out_channels, **kwargs)
        self.conv2 = _ICR1x1Conv(in_channels, out_channels, **kwargs)
        self.conv3 = _ICR1x1Conv(in_channels, out_channels, **kwargs)
        self.conv4 = _ICR1x1Conv(in_channels, out_channels, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode='bilinear', align_corners=True)
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)

# ============= ASPP =============
class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out

class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs, **kwargs):
        super(_ASPP, self).__init__()
        out_channels = 512
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x
