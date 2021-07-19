import torch
import numpy as np
from cvnn_modules import ComplexConv2d, RealConv2d, ComplexFC, RealFC, identity, group_sum
import torch.nn.functional as F
from funcs import SinFunc

def useComplexConv(useComp, in_channels, out_channels, kernel_size, padding, stride=1):
    layer = ComplexConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding) \
         if useComp else \
             RealConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    return layer

def useComplexFC(useComp, in_features, out_features, bias=True):
    layer = ComplexFC(in_features, out_features, bias=bias) \
        if useComp else \
            RealFC(in_features, out_features, bias=bias)
    return layer

SinFunc = SinFunc.apply

# Class definition for abstract gradient regularized networks. In this version of the code, gradient regularization is
# isolated in the training section
class GradRegNet(torch.nn.Module):

    def __init__(self, useComp=True, use_sin=False):
        super(GradRegNet, self).__init__()
        self.useComp = useComp
        self.sc = np.pi if self.useComp else 1.0
        self.use_sin = use_sin

    def forward(self, x):
        if self.use_sin:
            x = SinFunc(x)
        x = self.sc * x
        return self.forward_impl(x)

    def forward_impl(self, x):
        raise NotImplementedError("GradRegNet forward impl not implemented!")

class MNISTNetV0(GradRegNet):

    def __init__(self, useComp):
        super(MNISTNetV0, self).__init__(useComp=useComp)
        self.actFunc = identity if self.useComp else F.relu
        self.conv1 = useComplexConv(self.useComp, 1, 32, 3, 1)
        self.pool = torch.nn.AvgPool2d(2)
        self.conv2 = useComplexConv(self.useComp, 32, 64, 3, 1)
        self.fc1 = useComplexFC(self.useComp, 7 * 7 * 64, 128)
        self.fc2 = useComplexFC(self.useComp, 128, 10)

    def forward_impl(self, x):
        x = self.pool(self.actFunc(self.conv1(x)))
        x = self.pool(self.actFunc(self.conv2(x)))
        x = x.view(-1, 7 * 7 * 64)
        x = self.actFunc(self.fc1(x))
        x = self.fc2(x)
        return x

class MNISTNetV1(GradRegNet):

    def __init__(self, useComp):
        super(MNISTNetV1, self).__init__(useComp=useComp)
        self.actFunc = identity if self.useComp else F.relu
        self.conv1 = useComplexConv(self.useComp, 1, 16, 3, 1)
        self.pool = torch.nn.AvgPool2d(2)
        self.conv2 = useComplexConv(self.useComp, 8, 32, 3, 1)
        self.fc1 = useComplexFC(self.useComp, 7 * 7 * 16, 128)
        self.fc2 = useComplexFC(self.useComp, 128, 10)

    def forward_impl(self, x):
        x = group_sum(self.actFunc(self.conv1(x)))
        x = self.pool(x)
        x = group_sum(self.actFunc(self.conv2(x)))
        x = self.pool(x)
        x = x.view(-1, 7 * 7 * 16)
        x = self.actFunc(self.fc1(x))
        x = self.fc2(x)
        return x

class MNISTNetV2(GradRegNet):

    def __init__(self, useComp):
        super(MNISTNetV2, self).__init__(useComp=useComp)
        self.actFunc = identity if self.useComp else F.relu
        self.conv1 = useComplexConv(self.useComp, 1, 64, 3, 1)
        self.pool = torch.nn.AvgPool2d(2)
        self.conv2 = useComplexConv(self.useComp, 8, 32, 3, 1)
        self.fc1 = useComplexFC(False, 7 * 7 * 32, 64)
        self.fc2 = useComplexFC(False, 64, 10)

    def forward_impl(self, x):
        x = group_sum(self.actFunc(self.conv1(x)), n_groups=8)
        x = self.pool(x)
        x = self.actFunc(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 7 * 7 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MNISTNetRDV(GradRegNet):

    def __init__(self, useComp):
        super(MNISTNetRDV, self).__init__(useComp=useComp)
        self.actFunc = identity if self.useComp else F.relu
        self.conv1 = useComplexConv(self.useComp, 1, 32, 5, 2)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = useComplexConv(self.useComp, 32, 64, 5, 2)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.fc1 = useComplexFC(False, 7 * 7 * 64, 1024)
        self.bnfc = torch.nn.BatchNorm1d(1024)
        self.drop = torch.nn.Dropout(p=0.5)
        self.fc2 = useComplexFC(False, 1024, 10)
        self.pool = torch.nn.AvgPool2d(2)

    def forward_impl(self, x):
        x = self.pool(self.actFunc(self.bn1(self.conv1(x))))
        x = self.pool(self.actFunc(self.bn2(self.conv2(x))))
        x = x.view(-1, 7 * 7 * 64)
        x = self.drop(F.relu(self.bnfc(self.fc1(x))))
        x = self.fc2(x)
        return x

class MNISTMLP(GradRegNet):

    def __init__(self, useComp):
        super(MNISTMLP, self).__init__(useComp=useComp)
        self.actFunc = identity if self.useComp else F.relu
        self.fc1 = useComplexFC(self.useComp, 28*28, 256)
        self.fc2 = useComplexFC(self.useComp, 256, 256)
        self.fc3 = useComplexFC(self.useComp, 256, 10)

    def forward_impl(self, x):
        x = x.view(-1, 28 * 28)
        x = self.actFunc(self.fc1(x))
        x = self.actFunc(self.fc2(x))
        x = self.fc3(x)
        return x

class CIFARNet1(GradRegNet):

    def __init__(self, useComp, alpha, beta):
        super(CIFARNet1, self).__init__(useComp=useComp, use_sin=False)
        self.actFunc = identity if self.useComp else F.relu
        self.conv1 = useComplexConv(self.useComp, 3, 32, 3, 1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.pool = torch.nn.AvgPool2d(2)
        self.conv2 = useComplexConv(self.useComp, 32, 64, 3, 1)
        self.conv3 = useComplexConv(self.useComp, 64, 128, 3, 1)
        self.fc1 = useComplexFC(self.useComp, 16*128, 256)
        self.fc2 = useComplexFC(self.useComp, 256, 10)

    def forward_impl(self, x):
        x = self.pool(self.actFunc(self.bn1(self.conv1(x))))
        x = self.pool(self.actFunc(self.bn2(self.conv2(x))))
        x = self.pool(self.actFunc(self.bn3(self.conv3(x))))
        x = x.view(-1, 16*128)
        x = self.actFunc(self.fc1(x))
        x = self.fc2(x)
        return x

class CIFARNet1_PP(GradRegNet):

    def __init__(self, useComp):
        super(CIFARNet1_PP, self).__init__(useComp=useComp, use_sin=useComp)
        self.actFunc = identity if self.useComp else F.relu
        l1_reg = 0
        self.conv1 = useComplexConv(self.useComp, 3, 32, 3, 1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.pool = torch.nn.AvgPool2d(2)
        self.conv2 = useComplexConv(self.useComp, 32, 64, 3, 1)
        self.conv3 = useComplexConv(self.useComp, 64, 128, 3, 1)
        self.fc1 = useComplexFC(self.useComp, 16*128, 256)
        self.fc2 = useComplexFC(self.useComp, 256, 10)

    def forward_impl(self, x):
        x = self.pool(self.actFunc(self.bn1(self.conv1(x))))
        x = self.pool(self.actFunc(self.bn2(self.conv2(x))))
        x = self.pool(self.actFunc(self.bn3(self.conv3(x))))
        x = x.view(-1, 16*128)
        x = self.actFunc(self.fc1(x))
        x = self.fc2(x)
        return x

class CIFARNet2(GradRegNet):

    def __init__(self, useComp):
        super(CIFARNet2, self).__init__(useComp=useComp, use_sin=False)
        self.actFunc = identity if self.useComp else F.relu
        self.conv1 = useComplexConv(self.useComp, 3, 64, 3, 1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.pool = torch.nn.AvgPool2d(2)
        self.conv2 = useComplexConv(self.useComp, 64, 128, 3, 1)
        self.conv3 = useComplexConv(self.useComp, 128, 256, 3, 1)
        self.fc1 = useComplexFC(self.useComp, 16*256, 256)
        self.fc2 = useComplexFC(self.useComp, 256, 10)

    def forward_impl(self, x):
        x = self.pool(self.actFunc(self.bn1(self.conv1(x))))
        x = self.pool(self.actFunc(self.bn2(self.conv2(x))))
        x = self.pool(self.actFunc(self.bn3(self.conv3(x))))
        x = x.view(-1, 16*256)
        x = self.actFunc(self.fc1(x))
        x = self.fc2(x)
        return x

arch_names = ['MNISTNetV0', 'MNISTNetV1', 'MNISTNetV2', 'MNISTNetRDV', 'MNISTMLP', 'CIFARNet1', 'CIFARNet1_PP', 'CIFARNet2']
arch_dict = {
    'MNISTNetV0': MNISTNetV0,
    'MNISTNetV1': MNISTNetV1,
    'MNISTNetV2': MNISTNetV2,
    'MNISTMLP': MNISTMLP,
    'CIFARNet1': CIFARNet1,
    'CIFARNet1_PP': CIFARNet1_PP,
    'CIFARNet2': CIFARNet2
}