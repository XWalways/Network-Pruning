import mxnet
from models import *
a = []
net = densenet()
for m in net._children.values():
    if isinstance(m, mxnet.gluon.nn.BatchNorm):
        a.append(m)
    else:
        for mm in m._children.values():
            if isinstance(mm, mxnet.gluon.nn.BatchNorm):
                a.append(mm)
            elif isinstance(mm, BasicBlock):
                for mmm in mm._children.values():
                    if isinstance(mmm, mxnet.gluon.nn.BatchNorm):
                        a.append(mmm)


print(a)
print(len(a))