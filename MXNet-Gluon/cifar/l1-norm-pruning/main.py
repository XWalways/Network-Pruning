#Baseline
import argparse
import numpy as np
import os
import shutil
import logging
import mxnet
import time
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
import random
import gluoncv as gcv
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs
from gluoncv.data import transforms as gcv_transforms
from mxboard import SummaryWriter

import models

os.environ['MXNET_SAFE_ACCUMULATION'] = '1'
#os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    #=========================model==============================================
    parser.add_argument('--arch', default='vgg', type=str,
                        help='architecture to use')
    parser.add_argument('--depth', default=16, type=int,
                        help='depth of the neural network')

    #========================dataset=============================================
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to use. options are cifar10 and cifar100. default is cifar10.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=8, type=int,
                        help='number of preprocessing workers')

    #========================training HPs========================================
    parser.add_argument('--random-seed', type=int, default=2,
                        help='random seed (default: 1)')
    parser.add_argument('--num-gpus', type=int, default=8,
                        help='number of gpus to use.')
    parser.add_argument('--num-epochs', type=int, default=160,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='period in epoch for learning rate decays. default is 0 (has no effect).')
    parser.add_argument('--lr-decay-epoch', type=str, default='80,120',
                        help='epochs at which learning rate decays. default is 80,120.')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are imperative, hybrid')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='training dtype')

    #==============================save&log&resume====================================
    parser.add_argument('--save-period', type=int, default=50,
                        help='period in epoch of model saving.')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='directory of saved logs')
    parser.add_argument('--resume', type=str,
                        help='resume training from the model')

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_args()

    batch_size = opt.batch_size
    num_gpus = opt.num_gpus
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    for ctx in context:
        mx.random.seed(seed_state=opt.random_seed, ctx=ctx)
    np.random.seed(opt.random_seed)
    random.seed(opt.random_seed)
    num_workers = opt.num_workers
    lr_decay = opt.lr_decay
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]
    save_period = opt.save_period
    log_dir = opt.log_dir
    save_dir = opt.save_dir
    makedirs(save_dir)
    makedirs(log_dir)

    if opt.dataset == 'cifar10':
        classes = 10
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
            batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

        val_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif opt.dataset == 'cifar100':
        classes = 100
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


    model = models.__dict__[opt.arch](dataset=opt.dataset, depth=opt.depth)
    model_name = opt.arch + '_' + str(opt.depth)
    if opt.resume:
        net.load_parameters(opt.resume, ctx=context)

    sw = SummaryWriter(logdir=log_dir, flush_secs=5, verbose=False)


    logging.basicConfig(level=logging.INFO)
    logging.info(opt)

    def test(ctx, val_data):
        metric = mx.metric.Accuracy()
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = [model(X.astype(opt.dtype, copy=False)) for X in data]
            metric.update(label, outputs)
        return metric.get()

    def train(epochs, ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        model.initialize(mx.init.Xavier(), ctx=ctx)
        trainer = gluon.Trainer(model.collect_params(), 'sgd',
                                {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum})
        metric = mx.metric.Accuracy()
        train_metric = mx.metric.Accuracy()
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
        iteration = 0
        lr_decay_count = 0

        best_val_score = 0

        for epoch in range(epochs):
            tic = time.time()
            btic = time.time()
            train_metric.reset()
            metric.reset()
            train_loss = 0
            num_batch = len(train_data)
            if epoch == lr_decay_epoch[lr_decay_count]:
                trainer.set_learning_rate(trainer.learning_rate * lr_decay)
                lr_decay_count += 1

            for i, batch in enumerate(train_data):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

                with ag.record():
                    output = [model(X.astype(opt.dtype, copy=False)) for X in data]
                    loss = [loss_fn(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(output, label)]
                for l in loss:
                    l.backward()
                trainer.step(batch_size)
                train_loss += sum([l.sum().asscalar() for l in loss])
                sw.add_scalar(tag='train_loss', value=train_loss / len(loss), global_step=iteration)

                train_metric.update(label, output)
                name, acc = train_metric.get()
                sw.add_scalar(tag='train_{}_curves'.format(name),
                              value=('train_{}_value'.format(name), acc),
                              global_step=iteration)

                if opt.log_interval and not (i + 1) % opt.log_interval:
                    logging.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f\tlr=%f' % (
                        epoch, i, batch_size * opt.log_interval / (time.time() - btic),
                        name, acc, trainer.learning_rate))
                    btic = time.time()

                iteration += 1
            if epoch == 0:
                sw.add_graph(model)
            train_loss /= batch_size * num_batch
            _, acc = train_metric.get()
            _, val_acc = test(ctx, val_data)
            sw.add_scalar(tag='val_acc_curves', value=('valid_acc_value', val_acc), global_step=epoch)

            if val_acc > best_val_score:
                best_val_score = val_acc
                model.save_parameters('%s/%.4f-%s-%s-%d-best.params' % (save_dir, best_val_score, opt.dataset, model_name, epoch))
                trainer.save_states('%s/%.4f-%s-%s-%d-best.states' % (save_dir, best_val_score, opt.dataset, model_name, epoch))

            logging.info('[Epoch %d] train=%f val=%f loss=%f time: %f' %
                         (epoch, acc, val_acc, train_loss, time.time() - tic))

            if save_period and save_dir and (epoch + 1) % save_period == 0:
                model.save_parameters('%s/%s-%s-%d.params' % (save_dir, opt.dataset, model_name, epoch))
                trainer.save_states('%s/%s-%s-%d.states'%(save_dir, opt.datset, model_name, epoch))

        if save_period and save_dir:
            model.save_parameters('%s/%s-%s-%d.params' % (save_dir, opt.dataset, model_name, epochs - 1))
            trainer.save_states('%s/%s-%s-%d.states' % (save_dir, opt.dataset, model_name, epochs - 1))

    if opt.mode == 'hybrid':
        model.hybridize()
    train(opt.num_epochs, context)

if __name__ == '__main__':
    main()










