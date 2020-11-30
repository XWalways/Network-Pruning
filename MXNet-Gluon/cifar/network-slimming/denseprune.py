import argparse
import numpy as np
import os
import random

import mxnet
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from gluoncv.utils import makedirs
from mxnet.gluon.data.vision import transforms
from compute_flops import print_model_param_flops

from gluoncv.data import transforms as gcv_transforms
from models import *

os.environ['MXNET_SAFE_ACCUMULATION'] = '1'
#os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

parser = argparse.ArgumentParser(description='MxNet Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--num-gpus', type=int, default=1,
                    help='number of gpus')
parser.add_argument('--random-seed', type=int, default=2,
                    help='random seed')
parser.add_argument('--num-workers', type=int, default=8,
                    help='number of workers')
parser.add_argument('--dtype', default='float32', type=str,
                    help='dtype of the model')
parser.add_argument('--depth', type=int, default=40,
                    help='depth of the resnet')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

args = parser.parse_args()

batch_size = args.test_batch_size
batch_size *= max(1, args.num_gpus)
context = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
for ctx in context:
    mx.random.seed(seed_state=args.random_seed, ctx=ctx)
np.random.seed(args.random_seed)
random.seed(args.random_seed)

model = densenet(depth=args.depth, dataset=args.dataset)
model_name = 'resnet' + '_' + str(args.depth)
if args.model:
    if os.path.isfile(args.model):
        model.load_parameters(args.model, ctx=context)
    print('Pre-processing Successful!')

total = 0

for m in model._children.values():
    if isinstance(m, nn.BatchNorm):
        total += m.gamma._data[0].shape[0]
    elif isinstance(m, Transition) or isinstance(m, nn.HybridSequential):
        for mm in m._children.values():
            if isinstance(mm, nn.BatchNorm):
                total += mm.gamma._data[0].shape[0]
            if isinstance(mm, BasicBlock):
                for mmm in mm._children.values():
                    if isinstance(mmm, nn.BatchNorm):
                        total += mm.gamma._data[0].shape[0]

bn = mxnet.ndarray.zeros(total)
index = 0
for m in model._children.values():
    if isinstance(m, nn.BatchNorm):
        size = m.gamma._data[0].shape[0]
        bn[index, (index+size)] = m.gamma._data[0].abs()
        index += size
    if isinstance(m, nn.HybridSequential) or isinstance(m, Transition):
        for mm in m._children.values():
            if isinstance(mm, nn.BatchNorm):
                size = mm.gamma._data[0].shape[0]
                bn[index:(index+size)] = mm.gamma._data[0].abs()
                index += size
            if isinstance(mm, BasicBlock):
                for mmm in mm._children.values():
                    if isinstance(mmm, nn.BatchNorm):
                        size = mmm.gamma._data[0].shape[0]
                        bn[index:(index + size)] = mmm.gamma._data[0].abs()
                        index += size

y = mx.ndarray.sort(bn)
thre_index = int(total*args.percent)
thre = y[thre_index]

pruned = 0
cfg = []
cfg_mask = []
for m in model._children.values():
    if isinstance(m, nn.BatchNorm):
        weight = m.gamma._data[0]
        bias = m.beta._data[0]
        weight_copy = m.gamma._data[0].abs().copy()
        mask = mxnet.ndarray.cast(mxnet.ndarray.greater(weight_copy, thre).copyto(mxnet.gpu()), dtype='float32')
        pruned = pruned + mask.shape[0] - mask.sum().asnumpy()
        m.gamma._data[0] = mxnet.ndarray.broadcast_mul(weight, mask)
        m.beat._data[0] = mxnet.ndarray.broadcast_mul(bias, mask)
        cfg.append(int(mask.sum().asnumpy()))
        cfg_mask.append(mask.copy())
    if isinstance(m, nn.MaxPool2D):
        cfg.append('M')
    if isinstance(m, nn.HybridSequential) or isinstance(m, Transition):
        for mm in m._children.values():
            if isinstance(mm, nn.BatchNorm):
                weight = mm.gamma._data[0]
                bias = mm.beta._data[0]
                weight_copy = mm.gamma._data[0].abs().copy()
                mask = mxnet.ndarray.cast(mxnet.ndarray.greater(weight_copy, thre).copyto(mxnet.gpu()),
                                          dtype='float32')
                pruned = pruned + mask.shape[0] - mask.sum().asnumpy()
                mm.gamma._data[0] = mxnet.ndarray.broadcast_mul(weight, mask)
                mm.beat._data[0] = mxnet.ndarray.broadcast_mul(bias, mask)
                cfg.append(int(mask.sum().asnumpy()))
                cfg_mask.append(mask.copy())
            if isinstance(mm, nn.MaxPool2D):
                cfg.append('M')
            if isinstance(mm, BasicBlock):
                for mmm in mm._children.values():
                    if isinstance(mmm, nn.BatchNorm):
                        weight = mmm.gamma._data[0]
                        bias = mmm.beta._data[0]
                        weight_copy = mmm.gamma._data[0].abs().copy()
                        mask = mxnet.ndarray.cast(mxnet.ndarray.greater(weight_copy, thre).copyto(mxnet.gpu()),
                                                  dtype='float32')
                        pruned = pruned + mask.shape[0] - mask.sum().asnumpy()
                        mmm.gamma._data[0] = mxnet.ndarray.broadcast_mul(weight, mask)
                        mmm.beat._data[0] = mxnet.ndarray.broadcast_mul(bias, mask)
                        cfg.append(int(mask.sum().asnumpy()))
                        cfg_mask.append(mask.copy())
                    if isinstance(mmm, nn.MaxPool2D):
                        cfg.append('M')


pruned_ratio = pruned / total
print('Pruned Done!')
def test(model):
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            gcv_transforms.RandomCrop(32, pad=4),
            transforms.RandomFlipLeftRight(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        train_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
            batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=args.num_workers)

        val_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            gcv_transforms.RandomCrop(32, pad=4),
            transforms.RandomFlipLeftRight(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
        ])
        train_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR100(train=True).transform_first(transform_train),
            batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

        val_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR100(train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)
    metric = mx.metric.Accuracy()
    metric.reset()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=context, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=context, batch_axis=0)
        outputs = [model(X.astype(args.dtype, copy=False)) for X in data]
        metric.update(label, outputs)
        _, acc = metric.get()
    print('\nTest-set Accuracy: ', acc)
    return acc

acc = test(model)
print("cfg:")
print(cfg)

newmodel = densenet(depth=args.depth, dataset=args.dataset, cfg=cfg)

num_parameters, flops = print_model_param_flops(newmodel, input_res=32)

savepath = os.path.join(args.save, "prune.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc))


layer_id_in_cfg = 0
start _mask = mxnet.ndarray.ones((3,))
end_mask = cfg_mask[layer_id_in_cfg]
first_conv = True

old_modules = []
new_modules = []

for m in model._children.values():
    if not isinstance(m, nn.HybridSequential) and not isinstance(m, Transition):
        old_modules.append(m)
    else:
        for mm in m._children.values():
            if not isinstance(mm, BasicBlock):
                old_modules.append(mm)
            else:
                for mmm in mm._children.values():
                    old_modules.append(mmm)

for m in newmodel._children.values():
    if not isinstance(m, nn.HybridSequential) and not isinstance(m, Transition):
        new_modules.append(m)
    else:
        for mm in m._children.values():
            if not isinstance(mm, BasicBlock):
                new_modules.append(mm)
            else:
                for mmm in mm._children.values():
                    new_modules.append(mmm)

for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    if isinstance(m0, nn.BatchNorm) and isinstance(m1, nn.BatchNorm):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.copyto(mxnet.cpu()).asnumpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        if isinstance(old_modules[layer_id+1], channel_selection) and isinstance(new_modules[layer_id+1], channel_selection):
            m1.gamma._data[0] = m0.gamma._data[0].copy()
            m1.beta._data[0] = m0.beta._data[0].copy()
            m1.running_mean._data[0] = m0.running_mean._data[0].copy()
            m1.running_var._data[0] = m0.running_var._data[0].copy()

            m2 = new_modules[layer_id + 1]
            indexes = m2.indexes._data[0]
            m2.indexes._data[0] = mxnet.nd.zeros_like(indexes)
            m2.indexes._data[0][idx1.tolist()] = 1.0

            layer_id_in_cfg += 1
            start_mask = end_mask.copy()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
            continue
    elif isinstance(m0, nn.Conv2D):
        if first_conv:
            m1.weight._data[0] = m0.weight._data[0].copy()
            first_conv = False
            continue
        if isinstance(old_modules[layer_id_in_cfg-1], channel_selection):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.copyto(mxnet.cpu()).asnumpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.copyto(mxnet.cpu()).asnumpy())))

            print('In shape: {:}, Out shape: {:}'.format(idx0.size, idx1.szie))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.sisze == 1:
                idx1 = np.resize(idx1, (1,))

            w1 = m0.weight._data[0][:, idx0.tolist(), :, :].copy()
            m1.weight._data[0] = w1.copy()
            continue
    elif isinstance(m0, nn.Dense):
        idx0 = np.squeeze(np.argwhere(np.argwhere(start_mask.copyto(mxnet.cpu().asnumpy()))))
        if idx0.szie == 1:
            idx0 = np.resize(idx0, (1,))

        m1.weight._data[0] = m0.weight._data[0][:, idx0].copy()
        m1.bias._data[0] = m0.bias._data[0].copy()

pruned_model = '%s/%s-%s-pruned.params' % (args.save, args.dataset, model_name)
mxnet.ndarray.save(pruned_model, params)
newmodel.collect_params().load(pruned_model, ctx=context)
acc = test(newmodel)

num_parameters, flops = print_model_param_flops(newmodel, input_res=32)




print('\nTest-set accuracy after pruning: ', acc)
print('\nThe Pruned Channel Config Which Will be Used in Finetuning or Retraining: ', cfg)



















