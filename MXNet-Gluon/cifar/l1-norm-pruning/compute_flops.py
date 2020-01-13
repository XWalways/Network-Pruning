import numpy as np
import mxnet
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

def print_model_param_flops(model, input_res=224, multiply_adds=True):
    list_conv_flops = []
    list_conv_params = []
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].shape
        output_channels, output_height, output_width = output[0].shape
        assert self._in_channels % self._kwargs['num_group'] == 0

        kernel_ops = self._kwargs['kernel'][0] * self._kwargs['kernel'][1] * (
                    self._in_channels // self._kwargs['num_group'])
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size *  (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_height * output_width

        list_conv_flops.append(flops)
        list_conv_params.append(params)

    list_dense_flops = []
    list_dense_params = []
    def dense_hook(self, input, output):
        batch_size = input[0].shape[0] if len(input[0].shape) == 2 else 1
        weight_ops = self.weight.shape[0] * self.weight.shape[1] * (2 if multiply_adds else 1)
        bias_ops = self.bias.shape[0] if self.bias is not None else 0

        flops = batch_size * (weight_ops + bias_ops)

        list_dense_flops.append(flops)
        list_dense_params.append(weight_ops + bias_ops)

    list_bn_flops = []
    list_bn_params = []
    def bn_hook(self, input, output):
        if len(input[0].shape) == 4:
            list_bn_flops.append(input[0].shape[0] * input[0].shape[1] * input[0].shape[2] * input[0].shape[3] * 2)
        elif len(input[0].shape) == 2:
            list_bn_flops.append(input[0].shape[0] * input[0].shape[1] * 2)
        list_bn_params.append(input[0].shape[1]*2)


    list_relu = []
    def relu_hook(self, input, output):
        if len(input[0].shape) == 4:
            list_relu.append(input[0].shape[0] * input[0].shape[1] * input[0].shape[2] * input[0].shape[3])
        elif len(input[0].shape) == 2:
            list_relu.append(input[0].shape[0] * input[0].shape[1])

    list_pooling = []
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].shape
        output_channels, output_height, output_width = output[0].shape

        kernel_ops = self._kwargs['kernel'][0] * self._kwargs['kernel'][1]
        bias_ops = 0
        params = 0

        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size
        list_pooling.append(flops)

    def get(model=model):
        for m in model._children.values():
            if isinstance(m, nn.Conv2D):
                m.register_forward_hook(conv_hook)
            if isinstance(m, nn.Dense):
                m.register_forward_hook(dense_hook)
            if isinstance(m, nn.BatchNorm):
                m.register_forward_hook(bn_hook)
            if isinstance(m, nn.Activation):
                m.register_forward_hook(relu_hook)
            if isinstance(m, nn.GlobalAvgPool2D) or isinstance(m, nn.MaxPool2D) or isinstance(m, nn.AvgPool2D):
                m.register_forward_hook(pooling_hook)
            if isinstance(m, nn.HybridSequential):
                get(m)

    get(model)
    input = nd.random.uniform(-1, 1, shape=(1, 3, input_res, input_res), ctx=mx.cpu(0))
    model.initialize(mxnet.init.Xavier(), ctx=mx.cpu(0))
    out = model(input)
    total_flops = (sum(list_conv_flops) + sum(list_dense_flops) + sum(list_bn_flops) + sum(list_relu) + sum(list_pooling))
    total_params = (sum(list_conv_params) + sum(list_dense_params) + sum(list_bn_params))

    print('Number of params: %.2fM' % (total_params / 1e6))

    print('\nNumber of FLOPs: %.5fG' % (total_flops / 3 / 1e9))

    return total_flops / 3, total_params



if __name__ == '__main__':
    import models
    net = models.vgg()
    #print(net)
    flops, _ = print_model_param_flops(net, input_res=32, multiply_adds=True)
    #print(flops)










