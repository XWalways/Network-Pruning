import math
import mxnet
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

__all__ = ['vgg']

defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class vgg(nn.HybridBlock):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self.feature = self._make_layers(cfg, True)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Dense(num_classes, in_units=cfg[-1])
        if init_weights:
            self._initialize()
    def _make_layers(self, cfg, batch_norm=False):
        with self.name_scope():
            layers = nn.HybridSequential()
        in_channels=3
        for v in cfg:
            if v == 'M':
                layers.add(nn.MaxPool2D(pool_size=2, strides=2))
            else:
                conv2d = nn.Conv2D(v, in_channels=in_channels, kernel_size=3, padding=1, use_bias=False)
                if batch_norm:
                    layers.add(conv2d)
                    layers.add(nn.BatchNorm(in_channels=v, momentum=0.1))
                    layers.add(nn.Activation('relu'))
                else:
                    layers.add(conv2d)
                    layers.add(nn.Activation('relu'))
                in_channels = v
        return layers

    def hybrid_forward(self, F, x):
        x = self.feature(x)
        x = nn.AvgPool2D(pool_size=2)(x)
        y = self.classifier(x)
        return y

    def _initialize(self, force_reinit=True, ctx=mxnet.cpu()):
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



