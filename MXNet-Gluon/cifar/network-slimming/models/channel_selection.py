import numpy as np
import mxnet as mx
import mxnet
from mxnet import gluon
from mxnet.gluon import nn

class channel_selection(nn.HybridBlock):
    #channel_selection mask
    def __init__(self, num_channels):
        super(channel_selection, self).__init__()
        self.indexes = self.params.get('cs_indexes', shape=(num_channels,), init=mx.init.Constant(1))

    def hybrid_forward(self, F, input_tensor, *args, **kwargs):
        selected_index = np.squeeze(np.argwhere(self.indexes.data().asnumpy()))
        if selected_index.size == 1:
            selected_index = np.reszie(selected_index, (1,))
        output = input_tensor[:, selected_index, :, :]
        return output


