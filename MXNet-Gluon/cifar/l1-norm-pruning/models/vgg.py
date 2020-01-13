import math
import mxnet
from mxnet import gluon
from mxnet.gluon import nn

__all__=['vgg']

defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class vgg(nn.HybridBlock):
    def __init__(self, dataset='cifar10', depth=19, cfg=None):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg

        self.feature = self.make_layers(cfg, True)
        self.pool = nn.GlobalAvgPool2D()

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100

        self.classifier = nn.HybridSequential()
        with self.classifier.name_scope():
            self.classifier.add(nn.Dense( 512, in_units=cfg[-1]))
            self.classifier.add(nn.BatchNorm(in_channels=512))
            self.classifier.add(nn.Activation('relu'))
            self.classifier.add(nn.Dense(num_classes, in_units=512))


    def make_layers(self, cfg, batch_norm=False):
        layers = nn.HybridSequential()

        in_channels = 3
        with layers.name_scope():
            for v in cfg:
                if v == 'M':
                    layers.add(nn.MaxPool2D(pool_size=2, strides=2))
                else:
                    if batch_norm:
                        layers.add(nn.Conv2D(v, in_channels=in_channels, kernel_size=3, padding=1, use_bias=False))
                        layers.add(nn.BatchNorm(in_channels=v))
                        layers.add(nn.Activation('relu'))
                    else:
                        layers.add(nn.Conv2D(v, in_channels=in_channels, kernel_size=3, padding=1, use_bias=False))
                        layers.add(nn.Activation('relu'))

                    in_channels = v

        return layers

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.feature(x)
        x = self.pool(x)
        y = self.classifier(x)
        return y


if __name__ == '__main__':
    net = vgg(dataset='cifar10', depth=16)
    net.hybridize()
    net.initialize(mxnet.init.Xavier(), ctx=mxnet.gpu(0))
    x=mxnet.ndarray.zeros((16, 3, 32, 32), ctx=mxnet.gpu(0))
    y = net(x)
    net.save_parameters('../vgg.params')
    print(y.shape)


