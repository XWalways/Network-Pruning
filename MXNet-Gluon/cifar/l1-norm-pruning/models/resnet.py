import mxnet
from mxnet import gluon
from mxnet.gluon import nn
import math
from functools import partial

__all__=['resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2D(out_planes, in_channels=in_planes, kernel_size=3, strides=stride,
                     padding=1, use_bias=False)


class BasicBlock(nn.HybridBlock):
    expansion=1
    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        # cfg should be a number in this case
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, cfg, stride)
        self.bn1 = nn.BatchNorm(in_channels=cfg)
        self.relu1 = nn.Activation('relu')
        self.conv2 = conv3x3(cfg, planes)
        self.bn2 = nn.BatchNorm(in_channels=planes)
        self.relu2 = nn.Activation('relu')
        self.pool = nn.AvgPool2D(pool_size=2, strides=2)
        self.downsample = downsample
        self.stride = stride

    def hybrid_forward(self, F, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu2(out)

        return out

def downsample_basic_block(x, planes):
    x = mxnet.ndarray.Pooling(x,kernel=(2,2), pool_type='avg', stride=(2,2), layout='NCHW')
    zero_pads = mxnet.ndarray.zeros((x.shape[0], planes-x.shape[1], x.shape[2], x.shape[3]))
    zero_pads = zero_pads.as_in_context(x.context).astype(x.dtype,copy=False)

    out = mxnet.ndarray.concat(x, zero_pads, dim=1)

    return out


class ResNet(nn.HybridBlock):
    def __init__(self, depth, dataset='cifar10', cfg=None):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = BasicBlock
        if cfg == None:
            cfg = [[16] * n, [32] * n, [64] * n]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.cfg = cfg

        self.inplanes = 16
        self.conv1 = nn.Conv2D(16, in_channels=3, kernel_size=3, padding=1,
                               use_bias=False)
        self.bn1 = nn.BatchNorm(in_channels=16)
        self.relu = nn.Activation('relu')
        self.layer1 = self._make_layer(block, 16, n, cfg=cfg[0:n])
        self.layer2 = self._make_layer(block, 32, n, cfg=cfg[n:2 * n], stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg=cfg[2 * n:3 * n], stride=2)
        self.avgpool = nn.GlobalAvgPool2D()

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100

        self.fc = nn.Dense(num_classes, in_units=64*block.expansion)

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = partial(downsample_basic_block, planes=planes*block.expansion)

        layers = nn.HybridSequential()
        with layers.name_scope():
            layers.add(block(self.inplanes, planes, cfg[0], stride, downsample))
            self.inplanes = planes*block.expansion
            for i in range(1, blocks):
                layers.add(block(self.inplanes, planes, cfg[i]))


        return layers

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x



def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

if __name__ == '__main__':
    net = resnet(depth=56)
    net.initialize(mxnet.init.Xavier())
    x=mxnet.ndarray.zeros((16, 3, 32, 32))
    y = net(x)
    print(y.shape)




