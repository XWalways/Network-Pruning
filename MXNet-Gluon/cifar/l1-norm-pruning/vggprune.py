import argparse
import numpy as np
import os
from mxnet import gluon
from mxnet.gluon import nn
import mxnet
import mxnet as mx
from gluoncv.utils import makedirs
from models import *
from mxnet.gluon.data.vision import transforms
from compute_flops import print_model_param_flops
import random
from gluoncv.data import transforms as gcv_transforms


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
parser.add_argument('--depth', type=int, default=16,
                    help='depth of the vgg')
parser.add_argument('--model', default='', type=str,
                    help='path to the model (default: none)')
parser.add_argument('--dtype', default='float32', type=str,
                    help='dtype of the model')
parser.add_argument('--save', default='.', type=str,
                    help='path to save pruned model (default: none)')
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
model = vgg(dataset=args.dataset, depth=args.depth)
model.hybridize()
model_name = 'vgg' + '_' + str(args.depth)

if args.model:
    if os.path.isfile(args.model):
        model.load_parameters(args.model, ctx=context)
    print('Pre-processing Successful!')

else:
    print("=> no params found")

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
    print('\nTestset Accuracy: ', acc)
    return acc

acc = test(model)
cfg = [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256]

cfg_mask = []
layer_id = 0

for m in model._children.values():
    if isinstance(m, nn.HybridSequential):
        for mm in m._children.values():
            if isinstance(mm, nn.Conv2D):
                out_channels = mm.weight.shape[0]
                if out_channels == cfg[layer_id]:
                    cfg_mask.append(mx.ndarray.ones(out_channels))
                    layer_id += 1
                    continue
                weight = mm.weight._data[0]
                weight_copy = weight.abs().asnumpy()
                L1_norm = np.sum(weight_copy, axis=(1,2,3))
                arg_max = np.argsort(L1_norm)
                arg_max_rev = arg_max[::-1][:cfg[layer_id]]
                assert arg_max_rev.size == cfg[layer_id], "size of arg_max_rev not correct"
                mask = mx.ndarray.zeros(out_channels)
                mask[arg_max_rev.tolist()] = 1
                cfg_mask.append(mask)
                layer_id += 1
            elif isinstance(mm, nn.MaxPool2D):
                layer_id += 1

newmodel = vgg(dataset=args.dataset, depth=args.depth, cfg=cfg)
print(newmodel)
start_mask = mxnet.ndarray.ones(3)
layer_id_in_cfg = 0
end_mask = cfg_mask[layer_id_in_cfg]
params={}

for [m0, m1] in zip(model._children.values(), newmodel._children.values()):
    for [mm0, mm1] in zip(m0._children.values(), m1._children.values()):
        if isinstance(mm0, nn.BatchNorm):
            if layer_id_in_cfg < len(cfg_mask) :
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.asnumpy())))
                if idx1 .size == 1:
                    idx1 = np.resize(idx1,(1,))
                params[mm1.gamma.name] = mm0.gamma._data[0][idx1.tolist()]
                params[mm1.beta.name] = mm0.beta._data[0][idx1.tolist()]
                params[mm1.running_mean.name] = mm0.running_mean._data[0][idx1.tolist()]
                params[mm1.running_var.name] = mm0.running_var._data[0][idx1.tolist()]
                layer_id_in_cfg += 1
                start_mask = end_mask
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:#
                params[mm1.gamma.name] = mm0.gamma._data[0]
                params[mm1.beta.name] = mm0.beta._data[0]
                params[mm1.running_mean.name] = mm0.running_mean._data[0]
                params[mm1.running_var.name] = mm0.running_var._data[0]
                layer_id_in_cfg += 1

        elif isinstance(mm0, nn.Conv2D):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.asnumpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.asnumpy())))
            print('\nIn shape {:d}, Out shape {:d}'.format(idx0.size, idx1.size))

            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = mm0.weight._data[0][:, idx0.tolist(), :, :]
            w1 = w1[idx1.tolist(), :, :, :]
            params[mm1.weight.name] = w1

        elif isinstance(mm0, nn.Dense):
            if layer_id_in_cfg == len(cfg_mask):
                idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask[-1].asnumpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                params[mm1.weight.name] = mm0.weight._data[0][:, idx0]
                params[mm1.bias.name] = mm0.bias._data[0]
                layer_id_in_cfg += 1
                continue
            params[mm1.weight.name] = mm0.weight._data[0]
            params[mm1.bias.name] = mm0.bias._data[0]


#print(params)
pruned_model = '%s/%s-%s-pruned.params' % (args.save, args.dataset, model_name)
mxnet.ndarray.save(pruned_model, params)
newmodel.collect_params().load(pruned_model, ctx=context)
acc = test(newmodel)

num_parameters, flops = print_model_param_flops(newmodel, input_res=32)

print('\nTest accuracy after pruning: ', acc)





