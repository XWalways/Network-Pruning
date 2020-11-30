import math
import mxnet
import mxnet as mx
from mxnet.gluon import nn
from .channel_selection import channel_selection

__all__ = ['resnet']

class Bottleneck(nn.HybridBlock):
    expansion = 4
    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm(in_channels=inplanes, momentum=0.1)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2D(cfg[1], in_channels=cfg[0], kernel_size=1, use_bias=False)
        self.bn2 = nn.BatchNorm(in_channels=cfg[1], momentum=0.1)
        self.conv2 = nn.Conv2D(cfg[2], in_channels=cfg[1], kernel_size=3, strides=stride, padding=1, use_bias=False)
        self.bn3 = nn.BatchNorm(in_channels=cfg[2], momentum=0.1)
        self.conv3 = nn.Conv2D(4*planes, in_channels=cfg[2], kernel_size=1, use_bias=False)
        self.relu = nn.Activation('relu')
        self.downsample = downsample
        self.stride = stride

    def hybrid_forward(self, F, x):
        residual = x
        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class resnet(nn.HybridBlock):
    def __init__(self, depth=164, dataset='cifar10', cfg=None):
        super(resnet, self).__init__()
        assert (depth-2) % 9 == 0, 'depth should be 9n+2'

        n = (depth - 2) // 9
        block = Bottleneck

        if cfg is None:
            cfg = [[16, 16, 16], [64, 16, 16] * (n - 1), [64, 32, 32], [128, 32, 32] * (n - 1), [128, 64, 64],
                   [256, 64, 64] * (n - 1), [256]]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.inplanes = 16
        self.conv1 = nn.Conv2D(16, in_channels=3, kernel_size=3, padding=1, use_bias=False)
        self.layer1 = self._make_layer(block, 16, n, cfg=cfg[0:3*n])
        self.layer2 = self._make_layer(block, 32, n, cfg=cfg[3*n:6*n], stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg=cfg[6*n:9*n], stride=2)

        self.bn = nn.BatchNorm(in_channels=64*block.expansion)
        self.select = channel_selection(64*block.expansion)
        self.relu = nn.Activation('relu')
        self.avgpool = nn.AvgPool2D(pool_size=8)

        if dataset == 'cifar10':
            self.fc = nn.Dense(10, in_units=cfg[-1])
        elif dataset == 'cifar100':
            self.fc = nn.Dense(100, in_units=cfg[-1])

    def _initialize(self, force_reinit=True, ctx=mx.cpu()):
        for k, v in self.collect_params().items():
            if 'conv' in k:
                if 'weight' in k:
                    v.initialize(mx.init.Normal(1.0 / v.shape[1]), force_reinit=force_reinit, ctx=ctx)
                if 'bias' in k:
                    v.initialize(mx.init.Constant(0), force_reinit=force_reinit, ctx=ctx)
            elif 'batchnorm' in k:
                if 'gamma' in k:
                    v.initialize(mx.init.Constant(1), force_reinit=force_reinit, ctx=ctx)
                if 'beta' in k:
                    v.initialize(mx.init.Constant(0.0001), force_reinit=force_reinit, ctx=ctx)
                if 'running' in k or 'moving' in k:
                    v.initialize(mx.init.Constant(0), force_reinit=force_reinit, ctx=ctx)
            elif 'dense' in k:
                v.initialize(mx.init.Normal(0.01), force_reinit=force_reinit, ctx=ctx)
                if 'bias' in k:
                    v.initialize(mx.init.Constant(0), force_reinit=force_reinit, ctx=ctx)

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            with self.name_scope():
                downsample = nn.HybridSequential()
                downsample.add(nn.Conv2D(planes*block.expansion, in_channels=self.inplanes, kernel_size=1, strides=stride, use_bias=False))
        with self.name_scope():
            layers = nn.HybridSequential()
            layers.add(block(self.inplanes, planes, cfg[0:3], stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.add(block(self.inplanes, planes, cfg[3*i:3*(i+1)]))
        return layers

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.select(x)
        s = self.relu(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x




