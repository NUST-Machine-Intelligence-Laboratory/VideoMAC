import torch
import torch.nn as nn

from .utils import (
    LayerNorm,
    MinkowskiLayerNorm
)
from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiGELU
)
from MinkowskiOps import (
    to_sparse,
)


class BasicBlock(nn.Module):
    """ Sparse Basic Block.
        modified
    """
    expansion: int = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, D=2):
        super().__init__()
        self.conv1 = MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=stride, bias=False, dimension=D)
        self.norm1 = MinkowskiLayerNorm(out_channels, eps=1e-6)
        self.conv2 = MinkowskiConvolution(out_channels, out_channels, kernel_size=3, bias=False, dimension=D)
        self.norm2 = MinkowskiLayerNorm(out_channels, 1e-6)

        self.downsample = downsample
        self.stride = stride
        self.act = MinkowskiGELU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)
        return out


class Bottleneck(nn.Module):
    """ Sparse Bottleneck Block
        modified
    """
    expansion: int = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, D=2):
        super().__init__()
        self.conv1 = MinkowskiConvolution(in_channels, out_channels, kernel_size=1, bias=False, dimension=D)
        self.norm1 = MinkowskiLayerNorm(out_channels, eps=1e-6)
        self.conv2 = MinkowskiConvolution(out_channels, out_channels, kernel_size=3, stride=stride, bias=False, dimension=D)
        self.norm2 = MinkowskiLayerNorm(out_channels, eps=1e-6)
        self.conv3 = MinkowskiConvolution(out_channels, out_channels * self.expansion, kernel_size=1, bias=False, dimension=D)
        self.norm3 = MinkowskiLayerNorm(out_channels * self.expansion, eps=1e-6)

        self.downsample = downsample
        self.stride = stride
        self.act = MinkowskiGELU()

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.norm3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)
        return out


class SparseResNet(nn.Module):
    """ Sparse ResNet
        modified
    """
    def __init__(self, block, layers, pretrained=False):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4),
            LayerNorm(64, eps=1e-6, data_format="channels_first")
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, out_channels, blocks, stride=1, D=2):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                MinkowskiConvolution(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False, dimension=D),
                MinkowskiLayerNorm(out_channels * block.expansion, eps=1e-6)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2) 
    
    def forward(self, x, mask):
        mask = self.upsample_mask(mask, 8)
        mask = mask.unsqueeze(1).type_as(x)

        x = self.stem(x)
        x *= (1. - mask)
        # sparse encoding
        x = to_sparse(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # densify
        x = x.dense()[0]
        return x
