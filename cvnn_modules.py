import torch
import numpy as np
from funcs import ComplexFC_Func, RealFC_Func, RealConv2d_Func, ComplexConv2d_Func

useOldInitFC = False
useOldInitConv = False

def group_sum(x, n_groups=2):
    gsize = x.shape[1]//n_groups
    x = sum([x[:, i*gsize:(i+1)*gsize] for i in range(n_groups)])
    return x

class IdentityAct(torch.autograd.Function):

    """
    Placeholder function object for dynamic PhasorConv / Non-PhasorConv networks
    """

    @staticmethod
    def forward(ctx, inp):
        return inp

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

identity = IdentityAct.apply

class SmoothConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode = 'zeros'):
        super(SmoothConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.sl = torch.zeros(1).to('cuda:0' if torch.cuda.is_available() else 'cpu')

    def reset_parameters(self):
        raise NotImplementedError("Reset Parameters has not yet been implemented!")

    def forward(self, x):
        return self.forward_impl(x)

    def forward_impl(self, x):
        raise NotImplementedError("Forward function not implemented!")

class ComplexConv2d(SmoothConv2d):
    """
    Class which inherits from torch.nn.Module and implements the Phase-to-Magnitude activation function
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode = 'zeros'):
        super(ComplexConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, padding_mode)
        # model parameters
        self.wR = torch.nn.Parameter(torch.Tensor(self.out_channels, self.in_channels//self.groups, self.kernel_size, self.kernel_size))
        self.wI = torch.nn.Parameter(torch.Tensor(self.out_channels, self.in_channels//self.groups, self.kernel_size, self.kernel_size))
        self.bR = torch.Tensor(self.out_channels)
        self.bI = torch.Tensor(self.out_channels)
        self.bR = torch.nn.Parameter(self.bR)
        self.bI = torch.nn.Parameter(self.bI)
        self.reset_parameters_old() if useOldInitConv else self.reset_parameters()

    def reset_parameters_old(self):
        init_constant = 1./(self.kernel_size**2 * self.in_channels)
        angles = torch.empty_like(self.wR).uniform_(0, 2*np.pi)
        self.wR.data = init_constant * angles.cos()
        self.wI.data = init_constant * angles.sin()
        b_angles = torch.empty_like(self.bR).uniform_(0, 2*np.pi)
        self.bR.data = init_constant * b_angles.cos()
        self.bI.data = init_constant * b_angles.sin()
        self.Y = None

    def reset_parameters(self):
        init_constant = 1./((self.kernel_size**2 * self.in_channels)**0.5)
        angles = torch.empty_like(self.wR).uniform_(0, 2*np.pi)
        self.wR.data = init_constant * angles.cos()
        self.wI.data = init_constant * angles.sin()
        b_angles = torch.empty_like(self.bR).uniform_(0, 2*np.pi)
        self.bR.data = init_constant * b_angles.cos()
        self.bI.data = init_constant * b_angles.sin()
        self.Y = None

    def forward_impl(self, x):
        return ComplexConv2d_Func.apply(x, self.wR, self.wI, self.bR, self.bI, self.stride, self.padding, self.dilation, self.groups)

class RealConv2d(SmoothConv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode = 'zeros'):
        super(RealConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, padding_mode)
        self.w = torch.nn.Parameter(torch.Tensor(self.out_channels, self.in_channels//self.groups, self.kernel_size, self.kernel_size))
        self.b = torch.nn.Parameter(torch.Tensor(self.out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init_constant = 1./np.sqrt(self.in_channels * self.kernel_size**2)
        self.w.data.uniform_(-init_constant, init_constant)
        self.b.data.zero_()

    def forward_impl(self, x):
        return RealConv2d_Func.apply(x, self.w, self.b, self.stride, self.padding, self.dilation, self.groups)

class SmoothFC(torch.nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(SmoothFC, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.bias = bias
        self.sl = torch.tensor(0.0).to("cuda:0")

    def forward(self, x):
        return self.forward_impl(x)

    def forward_impl(self, x):
        raise NotImplementedError("Forward impl function not implemented!")


class RealFC(SmoothFC):
    def __init__(self, input_features, output_features, bias=True):
        super(RealFC, self).__init__(input_features, output_features, bias)
        self.w = torch.nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.b = torch.nn.Parameter(torch.Tensor(output_features))
        self.out = None
        self.reset_parameters()

    def reset_parameters(self):
        init_constant = 1./np.sqrt(self.input_features)
        self.w.data.uniform_(-init_constant, init_constant)
        if self.bias:
            self.b.data.uniform_(-init_constant, init_constant)

    def forward_impl(self, x):
        return RealFC_Func.apply(x, self.w, self.b)


class ComplexFC(SmoothFC):
    """
    Class definition for fully-connected Phasor module
    Code extended from https://pytorch.org/docs/1.1.0/notes/extending.html
    """
    def __init__(self, input_features, output_features, bias=True):
        super(ComplexFC, self).__init__(input_features, output_features, bias)
        self.weightsR = torch.nn.Parameter(torch.Tensor(output_features, input_features))
        self.weightsI = torch.nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.biasR = torch.nn.Parameter(torch.Tensor(output_features))
            self.biasI = torch.nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('biasR', None)
            self.register_parameter('biasI', None)
        self.reset_parameters_old() if useOldInitFC else self.reset_parameters()


    def reset_parameters_old(self):
        init_constant = 1./np.sqrt(self.input_features)
        self.weightsR.data.uniform_(-init_constant, init_constant)
        self.weightsI.data.uniform_(-init_constant, init_constant)
        if self.bias:
            self.biasR.data.uniform_(-init_constant, init_constant)
            self.biasI.data.uniform_(-init_constant, init_constant)

    def reset_parameters(self):
        init_constant = 1./np.sqrt(self.input_features)
        angles = torch.empty_like(self.weightsR).uniform_(0, 2*np.pi)
        self.weightsR.data = init_constant * angles.cos()
        self.weightsI.data = init_constant * angles.sin()
        if self.bias:
            b_angles = torch.empty_like(self.biasR).uniform_(0, 2*np.pi)
            self.biasR.data = init_constant * b_angles.cos()
            self.biasI.data = init_constant * b_angles.sin()

    def forward_impl(self, x):
        return ComplexFC_Func.apply(x, self.weightsR, self.weightsI, self.biasR, self.biasI)