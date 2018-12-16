import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def conv1x3x3(in_planes, out_planes, stride=(1,1,1)):
    """1x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,3,3), stride=stride,
                     padding=(0,1,1), bias=False)

def conv3x1x1(in_planes, out_planes, stride=(1,1,1)):
    """3x1x1 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3,1,1), stride=stride,
                     padding=(1,0,0), bias=False)

def conv1x1x1(in_planes, out_planes, stride=(1,1,1)):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,1), stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=(1,1,1), downsample=None, tempo_conv=False,):
        super(Bottleneck, self).__init__()
        if tempo_conv:
            self.conv1 = conv3x1x1(inplanes, planes)
        else:
            self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv1x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SlowPath(nn.Module):
    def __init__(self, inp_shape, temporal_stride):
        super(SlowPath, self).__init__()
        self.inplanes = 64
        self.temporal_stride = temporal_stride
        self.inp_shape = inp_shape
        self.sample_ids = torch.tensor([x for x in range(0, inp_shape[1], temporal_stride)])

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=(1,2,2))
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=(1,2,2), tempo_conv=True)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=(1,2,2), tempo_conv=True)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))


    def forward(self, x):
        x = torch.index_select(x, 2, self.sample_ids)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=(1,1,1), tempo_conv=False):
        downsample = None
        if stride[1] != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, tempo_conv))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

class FastPath(nn.Module):
    def __init__(self, inp_shape, temporal_stride):
        super(FastPath, self).__init__()
        self.inplanes = 8
        self.temporal_stride = temporal_stride
        self.inp_shape = inp_shape
        self.sample_ids = torch.tensor([x for x in range(0, inp_shape[1], temporal_stride)])

        self.conv1 = nn.Conv3d(3, 8, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3))
        self.bn1 = nn.BatchNorm3d(8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        self.layer1 = self._make_layer(Bottleneck, 8, 3, tempo_conv=True)
        self.layer2 = self._make_layer(Bottleneck, 16, 4, stride=(1,2,2), tempo_conv=True)
        self.layer3 = self._make_layer(Bottleneck, 32, 6, stride=(1,2,2), tempo_conv=True)
        self.layer4 = self._make_layer(Bottleneck, 64, 3, stride=(1,2,2), tempo_conv=True)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))


    def forward(self, x):
        x = torch.index_select(x, 2, self.sample_ids)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=(1,1,1), tempo_conv=False):
        downsample = None
        if stride[1] != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, tempo_conv))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
