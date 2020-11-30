import mxnet
import math
from mxnet import gluon
from mxnet.gluon import nn
import mxnet.ndarray as F
from .channel_selection import channel_selection

__all__ = ['densenet', 'BasicBlock', 'Transition']

#cfg is about network structure
cfg = []
"""
densenet with basic block.
"""
class BasicBlock(nn.HybridBlock):
    def __init__(self, inplanes, cfg, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()
        planes = expansion * inplanes
        self.bn1 = nn.BatchNorm(in_channels=inplanes, momentum=0.1)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2D(growthRate, in_channels=cfg, kernel_size=3, padding=1, use_bias=False)
        self.relu = nn.Activation('relu')
        self.dropRate = dropRate

    def hybrid_forward(self, F, x):
        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate)
        out = F.concat(x, out, dim=1)
        return out

class Transition(nn.HybridBlock):
    def __init__(self, inplanes, outplanes, cfg):
        super(Transition, self).__init__()

        self.bn1 = nn.BatchNorm(in_channels=inplanes, momentum=0.1)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2D(outplanes, in_channels=cfg, kernel_size=1, use_bias=False)
        self.relu = nn.Activation('relu')
        self.pool = nn.AvgPool2D(2)

    def hybrid_forward(self, F, x):
        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.pool(out)
        return out

class densenet(nn.HybridBlock):
    def __init__(self, depth=40, dropRate=0, dataset='cifar10', growthRate=12, compressionRate=1, cfg=None):
        super(densenet, self).__init__()

        assert (depth-4) % 3 == 0, 'depth should be 3n+4'

        n = (depth-4) // 3

        block = BasicBlock

        self.growthRate = growthRate
        self.dropRate = dropRate

        if cfg == None:
            cfg = []
            start = growthRate*2
            for i in range(3):
                cfg.append([start+12*i for i in range(n+1)])
                start += growthRate*12
            cfg = [item for sub_list in cfg for item in sub_list]

        assert len(cfg) == 3*n +3, 'length of cfg should be 3n+3'

        self.inplanes = growthRate*2
        self.conv1 = nn.Conv2D(self.inplanes, kernel_size=3, in_channels=3, padding=1, use_bias=False)
        self.dense1 = self._make_denseblock(block, n , cfg[0:n])
        self.trans1 = self._make_transition(compressionRate, cfg[n])
        self.dense2 = self._make_denseblock(block, n, cfg[n+1:2*n+1])
        self.trans2 = self._make_transition(compressionRate, cfg[2*n+1])
        self.dense3 = self._make_denseblock(block, n, cfg[2*n+2:3*n+2])

        self.bn = nn.BatchNorm(in_channels=self.inplanes, momentum=0.1)
        self.select = channel_selection(self.inplanes)
        self.relu = nn.Activation('relu')
        self.avgpool = nn.AvgPool2D(8)

        if dataset == 'cifar10':
            self.fc = nn.Dense(10, in_units=cfg[-1])
        elif dataset == 'cifar100':
            self.fc = nn.Dense(100, in_units=cfg[-1])

    def _initialize(self, force_reinit=True, ctx=mxnet.cpu()):
        for k, v in self.collect_params().items():
            if 'conv' in k:
                if 'weight' in k:
                    v.initialize(mxnet.init.Normal(1.0 / v.shape[1]), force_reinit=force_reinit, ctx=ctx)
                if 'bias' in k:
                    v.initialize(mxnet.init.Constant(0), force_reinit=force_reinit, ctx=ctx)
            elif 'batchnorm' in k:
                if 'gamma' in k:
                    v.initialize(mxnet.init.Constant(1), force_reinit=force_reinit, ctx=ctx)
                if 'beta' in k:
                    v.initialize(mxnet.init.Constant(0.0001), force_reinit=force_reinit, ctx=ctx)
                if 'running' in k or 'moving' in k:
                    v.initialize(mxnet.init.Constant(0), force_reinit=force_reinit, ctx=ctx)
            elif 'dense' in k:
                v.initialize(mxnet.init.Normal(0.01), force_reinit=force_reinit, ctx=ctx)
                if 'bias' in k:
                    v.initialize(mxnet.init.Constant(0), force_reinit=force_reinit, ctx=ctx)

    def _make_denseblock(self, block, blocks, cfg):
        with self.name_scope():
            layers = nn.HybridSequential()
        assert blocks == len(cfg), 'length of the cfg is not right'
        for i in range(blocks):
            layers.add(block(self.inplanes, cfg=cfg[i], growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate
        return layers
    def _make_transition(self, compressionRate, cfg):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes, cfg)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)

        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x
