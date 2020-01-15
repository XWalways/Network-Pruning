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
# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256,
                    help='input batch size for testing (default: 256)')
parser.add_argument('--num-gpus', type=int, default=8,
                    help='number of gpus')
parser.add_argument('--random-seed', type=int, default=2,
                    help='random seed')
parser.add_argument('--num-workers', type=int, default=8,
                    help='number of workers')
parser.add_argument('--depth', type=int, default=56,
                    help='depth of the vgg')
parser.add_argument('--model', default='', type=str,
                    help='path to the model (default: none)')
parser.add_argument('--dtype', default='float32', type=str,
                    help='dtype of the model')
parser.add_argument('--save', default='.', type=str,
                    help='path to save pruned model (default: none)')
parser.add_argument('-v', default='A', type=str,
                    help='version of the model')
args = parser.parse_args()

batch_size = args.test_batch_size
batch_size *= max(1, args.num_gpus)
context = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
for ctx in context:
    mx.random.seed(seed_state=args.random_seed, ctx=ctx)
np.random.seed(args.random_seed)
random.seed(args.random_seed)

if not os.path.exists(args.save):
    makedirs(args.save)
model = resnet(depth=args.depth, dataset=args.dataset)

model_name = 'resnet56' + '_' + str(args.depth)

if args.model:
    if os.path.isfile(args.model):
        model.load_parameters(args.model, ctx=context)
    print('Pre-processing Successful!')

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

skip = {
    'A': [16, 20, 38, 54],
    'B': [16, 18, 20, 34, 38, 54],
}

prune_prob = {
    'A': [0.1, 0.1, 0.1],
    'B': [0.6, 0.3, 0.1],
}

layer_id = 1
cfg = []
cfg_mask = []

for m in model._children.values():
    if isinstance(m, nn.HybridSequential):
        for mm in m._children.values():
            if isinstance(mm, nn.Conv2D):
                out_channels = mm.weight.shape[0]
                if layer_id in skip[args.v]:
                    cfg_mask.append(mx.ndarray.ones(out_channels))
                    layer_id += 1
                    continue
                if layer_id % 2 == 0:
                    if layer_id <= 18:
                        stage = 0
                    elif layer_id <= 36:
                        stage = 1
                    else:
                        stage = 2
                    prune_prob_stage = prune_prob[args.v][stage]
                    weight = mm.weight._data[0]
                    weight_copy = weight.abs().asnumpy()
                    L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
                    num_keep = int(out_channels * (1 - prune_prob_stage))
                    arg_max = np.argsort(L1_norm)
                    arg_max_rev = arg_max[::-1][:num_keep]
                    mask = mxnet.ndarray.zeros(out_channels)
                    mask[arg_max_rev.tolist()] = 1
                    cfg_mask.append(mask)
                    cfg.append(num_keep)
                    layer_id += 1
                    continue
                layer_id += 1

newmodel = resnet(dataset=args.dataset, depth=args.depth, cfg=cfg)
print(newmodel)
start_mask = torch.ones(3)
layer_id_in_cfg = 0
conv_count = 1
params={}

for [m0, m1] in zip(model._children.values(), newmodel._children.values()):
    for [mm0, mm1] in zip(m0._children.values(), m1._children.values()):
        if isinstance(mm0, nn.Conv2D):
            if conv_count == 1:
                params[mm1.weight.name] = mm0.weight._data[0]
                continue
            if conv_count % 2 == 0:
                mask = cfg_mask[layer_id_in_cfg]
                idx = np.squeeze(np.argwhere(np.asarray(mask.asnumpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = mm0.weight._data[0][idx.tolist(), :, :, :]
                params[mm1.weight.name] = w
                layer_id_in_cfg += 1
                conv_count += 1
                continue
            if conv_count % 2 == 1:
                mask = cfg_mask[layer_id_in_cfg - 1]
                idx = np.squeeze(np.argwhere(np.asarray(mask.asnumpy())))
                if idx.size == 1:
                    idx = np.resize(idx, (1,))
                w = mm0.weight._data[:, idx.tolist(), :, :]
                params[mm1.weight.name] = w
                conv_count += 1
                continue
            elif isinstance(mm0, nn.BatchNorm):
                if conv_count % 2 == 1:
                    mask = cfg_mask[layer_id_in_cfg-1]
                    idx = np.squeeze(np.argwhere(np.asarray(mask.asnumpy())))
                    if idx.size == 1:
                        idx = np.resize(idx, (1,))

                    params[mm1.gamma.name] = mm0.gamma._data[0][idx.tolist()]
                    params[mm1.beta.name] = mm0.beta._data[0][idx.tolist()]
                    params[mm1.running_mean.name] = mm0.running_mean._data[0][idx.tolist()]
                    params[mm1.running_var.name] = mm0.running_var._data[0][idx.tolist()]
                    continue

                params[mm1.gamma.name] = mm0.gamma._data[0]
                params[mm1.beta.name] = mm0.beta._data[0]
                params[mm1.running_mean.name] = mm0.running_mean._data[0]
                params[mm1.running_var.name] = mm0.running_var._data[0]

            elif isinstance(mm0, nn.Dense):
                params[mm1.weight.name] = mm0.weight._data[0]
                params[mm1.bias.name] = mm0.bias._data[0]

pruned_model = '%s/%s-%s-pruned.params' % (args.save, args.dataset, model_name)
mxnet.ndarray.save(pruned_model, params)
newmodel.collect_params().load(pruned_model, ctx=context)
acc = test(newmodel)

num_parameters, flops = print_model_param_flops(newmodel, input_res=32)

print('\nTest-set accuracy after pruning: ', acc)

