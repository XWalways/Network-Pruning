# ImageNet
This directory contains a pytorch implementation of the ImageNet experiments for six pruning methods:  

1. [L1-norm based channel pruning](https://arxiv.org/abs/1608.08710)
2. [ThiNet](https://arxiv.org/abs/1707.06342)
3. [Regression based feature reconstruction](https://arxiv.org/abs/1707.06168)
4. [Network Slimming](https://arxiv.org/abs/1708.06519)
5. [Sparse Structure Selection](https://arxiv.org/abs/1707.01213)
6. [Non-structured weight-level pruning](https://arxiv.org/abs/1506.02626)

## Implementation
We use the [official Gluoncv ImageNet training code](https://gluon-cv.mxnet.io/_downloads/6ecf3a8b8036c8af2c65a18f473f1acb/train_imagenet.py).

## Dependencies
mxnet-cu101mkl-1.5.1 or mxnet-cu100mkl-1.5.1 + gluoncv-0.5.0
