import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import BaseModel
from timm.models.layers import DropPath, create_attn

# from regularization.shakedrop import ShakeDrop
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        stochastic_depth=0.0,
        apply_se=False,
        p_shakedrop=None,
        dims=2,
    ):
        super(BasicBlock, self).__init__()
        if dims == 1:
            self.conv1 = nn.Conv1d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm1d(planes)
            self.conv2 = nn.Conv1d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm1d(planes)
        elif dims == 2:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(planes)
        elif dims == 3:
            self.conv1 = nn.Conv3d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm3d(planes)
            self.conv2 = nn.Conv3d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm3d(planes)

        # Stochastic Depth
        self.drop_path = DropPath(drop_prob=stochastic_depth)
        # Squeeze and Excitation
        self.apply_se = apply_se
        outplanes = planes * self.expansion
        # self.se = create_attn('se', outplanes, reduction_ratio=0.25)  # ratio from https://arxiv.org/pdf/2103.07579.pdf
        # Shakedrop
        self.shake_drop = None  # ShakeDrop(p_shakedrop) if p_shakedrop else None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if dims == 1:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm1d(self.expansion * planes),
                )
            elif dims == 2:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            elif dims == 3:
                self.shortcut = nn.Sequential(
                    nn.Conv3d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        if self.apply_se:
            out = self.se(out)
        out = self.drop_path(out)
        if self.shake_drop:
            out = self.shake_drop(out)

        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        stochastic_depth=0.0,
        apply_se=False,
        p_shakedrop=None,
        dims=2,
    ):
        super(Bottleneck, self).__init__()
        if dims == 1:
            self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm1d(planes)
            self.conv2 = nn.Conv1d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm1d(planes)
            self.conv3 = nn.Conv1d(
                planes, self.expansion * planes, kernel_size=1, bias=False
            )
            self.bn3 = nn.BatchNorm1d(self.expansion * planes)
        elif dims == 2:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(
                planes, self.expansion * planes, kernel_size=1, bias=False
            )
            self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        elif dims == 3:
            self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
            self.conv2 = nn.Conv3d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm3d(planes)
            self.conv3 = nn.Conv3d(
                planes, self.expansion * planes, kernel_size=1, bias=False
            )
            self.bn3 = nn.BatchNorm3d(self.expansion * planes)

        # Stochastic Depth
        self.drop_path = DropPath(drop_prob=stochastic_depth)
        # Squeeze and Excitation
        self.apply_se = apply_se
        outplanes = planes * self.expansion
        # self.se = create_attn('se', outplanes, reduction_ratio=0.25)  # ratio from https://arxiv.org/pdf/2103.07579.pdf
        # Shakedrop
        self.shake_drop = None  # ShakeDrop(p_shakedrop) if p_shakedrop else None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if dims == 1:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm1d(self.expansion * planes),
                )
            elif dims == 2:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            elif dims == 3:
                self.shortcut = nn.Sequential(
                    nn.Conv3d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = F.relu(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))

        if self.apply_se:
            out = self.se(out)
        out = self.drop_path(out)
        if self.shake_drop:
            out = self.shake_drop(out)

        out = out + self.shortcut(
            x
        )  # shakedrop backward function does not allow += notation
        out = F.relu(out)

        return out


class ResNet(BaseModel):
    def __init__(self, block, num_blocks, num_classes=10, hypparams={}):
        super(ResNet, self).__init__(hypparams)

        self.dims = hypparams["dims"]

        self.in_planes = 64

        # shakedrop
        n = int(np.sum(num_blocks))
        self.u_idx = 0
        if self.apply_shakedrop:
            self.ps_shakedrop = [1 - (1.0 - (0.5 / n) * (i + 1)) for i in range(n)]
        else:
            self.ps_shakedrop = [None for _ in range(n)]

        if self.dims == 1:
            self.conv1 = nn.Conv1d(
                4, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm1d(64)
            self.adaptivepool = nn.AdaptiveAvgPool1d(1)
        elif self.dims == 2:
            self.conv1 = nn.Conv2d(
                4, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64)
            self.adaptivepool = nn.AdaptiveAvgPool2d(1)
        elif self.dims == 3:
            self.conv1 = nn.Conv3d(
                4, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm3d(64)
            self.adaptivepool = nn.AdaptiveAvgPool3d(1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            # print(self.ps_shakedrop[self.u_idx])
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    self.stochastic_depth,
                    self.se,
                    self.ps_shakedrop[self.u_idx],
                    self.dims,
                )
            )
            self.in_planes = planes * block.expansion
            self.u_idx += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = self.adaptivepool(out)

        out = F.dropout(out, p=self.resnet_dropout, training=self.training)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes, hypparams):
    if not hypparams["bottleneck"]:
        return ResNet(
            BasicBlock, [2, 2, 2, 2], num_classes=num_classes, hypparams=hypparams
        )


def ResNet34(num_classes, hypparams):
    if not hypparams["bottleneck"]:
        return ResNet(
            BasicBlock, [3, 4, 6, 3], num_classes=num_classes, hypparams=hypparams
        )


def ResNet50(num_classes, hypparams):
    if hypparams["bottleneck"]:
        return ResNet(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, hypparams=hypparams
        )
    else:
        return ResNet(
            BasicBlock, [4, 6, 10, 5], num_classes=num_classes, hypparams=hypparams
        )


def ResNet101(num_classes, hypparams):
    if hypparams["bottleneck"]:
        return ResNet(
            Bottleneck, [3, 4, 23, 3], num_classes=num_classes, hypparams=hypparams
        )


def ResNet110(num_classes, hypparams):
    if hypparams["bottleneck"]:
        return ResNet(
            Bottleneck, [12, 12, 12, 12], num_classes=num_classes, hypparams=hypparams
        )


def ResNet152(num_classes, hypparams):
    if hypparams["bottleneck"]:
        return ResNet(
            Bottleneck, [3, 8, 36, 3], num_classes=num_classes, hypparams=hypparams
        )
    else:
        return ResNet(
            BasicBlock, [4, 13, 55, 4], num_classes=num_classes, hypparams=hypparams
        )
