import torch
import torch.nn.functional as F
from torch.nn.grad import conv2d_weight as c2d_w
from torch.nn.grad import conv2d_input as c2d_inp
import numpy as np

class SinFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return inp - ((2 * np.pi * inp).sin() / (2 * np.pi))

    @staticmethod
    def backward(ctx, grad_output):
        grad_inp = None
        inp, = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_inp = (1. - (2 * np.pi * inp).cos()) * grad_output
        return grad_inp


class ComplexFC_Func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, wR, wI, bR=None, bI=None):
        xR = torch.cos(inp)
        xI = torch.sin(inp)
        zR = xR.mm(wR.t()) - (xI.mm(wI.t()))
        zI = xI.mm(wR.t()) + (xR.mm(wI.t()))
        if bR is not None and bI is not None:
            zR = zR.add(bR.unsqueeze(0).expand_as(zR))
            zI = zI.add(bI.unsqueeze(0).expand_as(zI))
        zM = (zR.square() + zI.square()).sqrt()
        ctx.save_for_backward(xR, xI, wR, wI, zR, zI, zM)
        return zM

    @staticmethod
    def backward(ctx, grad_output):
        xR, xI, wR, wI, zR, zI, zM = ctx.saved_tensors
        grad_inp, grad_wR, grad_wI, grad_bR, grad_bI = \
            ComplexFC_Func_Backward.apply(ctx, grad_output, xR, xI, wR, wI, zR, zI, zM)
        return grad_inp, grad_wR, grad_wI, grad_bR, grad_bI

class ComplexFC_Func_Backward(torch.autograd.Function):

    @staticmethod
    def forward(ctx_fc_func_back, ctx, grad_output, xR, xI, wR, wI, zR, zI, zM):
        grad_inp = grad_wR = grad_wI = grad_bR = grad_bI = None
        zR_zM = zR.div(zM)
        zI_zM = zI.div(zM)
        zR_zM_g = zR_zM.mul(grad_output)
        zI_zM_g = zI_zM.mul(grad_output)
        df_wR = zR_zM.unsqueeze(2).bmm(xR.unsqueeze(1)) + (zI_zM.unsqueeze(2).bmm(xI.unsqueeze(1)))
        df_wI = zI_zM.unsqueeze(2).bmm(xR.unsqueeze(1)) - (zR_zM.unsqueeze(2).bmm(xI.unsqueeze(1)))
        J = (df_wI.mul(wR) - (df_wR.mul(wI)))
        if ctx.needs_input_grad[3] and ctx.needs_input_grad[4]:
            grad_bR = zR_zM_g.sum(0)
            grad_bI = zI_zM_g.sum(0)
        if ctx.needs_input_grad[0]:
            grad_inp = J.transpose(1, 2).bmm(grad_output.unsqueeze(2)).squeeze(2)
        if ctx.needs_input_grad[1] and ctx.needs_input_grad[2]:  
            grad_wR = (df_wR).mul(grad_output.unsqueeze(2))
            grad_wI = (df_wI).mul(grad_output.unsqueeze(2))
        ctx_fc_func_back.save_for_backward(grad_output, df_wR, df_wI, J)
        ctx_fc_func_back.mark_non_differentiable(grad_wR, grad_wI, grad_bR, grad_bI)
        return grad_inp, grad_wR, grad_wI, grad_bR, grad_bI

    @staticmethod
    def backward(ctx, grad_inp_g, grad_wR_g, grad_wI_g, grad_bR_g, grad_bI_g):
        # only calculate gradient for grad_output_g, grad_wR, grad_wI
        grad_ctx = grad_output_g = grad_xR = grad_xI = grad_wR = grad_wI = grad_zR = grad_zI = grad_zM = grad_bR = grad_bI = None
        grad_output, df_wR, df_wI, J = ctx.saved_tensors
        o_prod = torch.bmm(grad_output.unsqueeze(2), grad_inp_g.unsqueeze(1))
        if ctx.needs_input_grad[1]: # grad_output
            grad_output_g = torch.bmm(J, grad_inp_g.unsqueeze(2)).squeeze()
        if ctx.needs_input_grad[4]:
            grad_wR = o_prod.mul(df_wI).sum(dim=0)
        if ctx.needs_input_grad[5]:
            grad_wI = o_prod.mul(df_wR.mul(-1.)).sum(dim=0)
        return grad_ctx, grad_output_g, grad_xR, grad_xI, grad_wR, grad_wI, grad_zR, grad_zI, grad_zM, grad_bR, grad_bI

class ComplexConv2d_Func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, wR, wI, bR=None, bI=None, stride=1, padding=0, dilation=1, groups=1):
        cp = (stride, padding, dilation, groups)
        xR = inp.cos()
        xI = inp.sin()
        zR = F.conv2d(xR, wR, None, *cp).sub_(F.conv2d(xI, wI, None, *cp))
        zI = F.conv2d(xI, wR, None, *cp).add_(F.conv2d(xR, wI, None, *cp))
        if bR is not None and bI is not None:
            zR = zR + bR.reshape(1, bR.shape[0], 1, 1).expand_as(zR)
            zI = zI + bI.reshape(1, bI.shape[0], 1, 1).expand_as(zI)
        zM = (zR.square() + zI.square()).sqrt_()
        ctx.save_for_backward(xR, xI, wR, wI, zR, zI, zM)
        ctx.cp = cp
        return zM

    @staticmethod
    def backward(ctx, grad_output):
        xR, xI, wR, wI, zR, zI, zM = ctx.saved_tensors
        grad_inp, grad_wR, grad_wI, grad_bR, grad_bI = \
            ComplexConv2d_Func_Backward.apply(ctx, grad_output, xR, xI, wR, wI, zR, zI, zM)
        return grad_inp, grad_wR, grad_wI, grad_bR, grad_bI, None, None, None, None

        

class ComplexConv2d_Func_Backward(torch.autograd.Function):

    @staticmethod
    def forward(ctx_conv_func_back, ctx, grad_output, xR, xI, wR, wI, zR, zI, zM):
        grad_inp = grad_wR = grad_wI = grad_bR = grad_bI = None
        cp = ctx.cp
        zR_zM = zR.div(zM)
        zI_zM = zI.div(zM)
        zR_zM_g = zR_zM.mul(grad_output)
        zI_zM_g = zI_zM.mul(grad_output)
        if ctx.needs_input_grad[3] and ctx.needs_input_grad[4]:
            grad_bR = zR_zM_g.sum(dim=(0, 2, 3))
            grad_bI = zI_zM_g.sum(dim=(0, 2, 3))
        if ctx.needs_input_grad[1] and ctx.needs_input_grad[2]:
            wM = (wR.square() + wI.square()).sqrt_()
            grad_wR = c2d_w(xR, wR.shape, zR_zM_g, *cp).add_(c2d_w(xI, wR.shape, zI_zM_g, *cp))
            grad_wI = c2d_w(xR, wI.shape, zI_zM_g, *cp).sub_(c2d_w(xI, wI.shape, zR_zM_g, *cp))
        if ctx.needs_input_grad[0]:
            grad_inp = xR.mul(c2d_inp(xR.shape, wR, zI_zM_g, *cp).sub_(c2d_inp(xR.shape, wI, zR_zM_g, *cp)))
            grad_inp.sub_(xI.mul(c2d_inp(xI.shape, wR, zR_zM_g, *cp).add_(c2d_inp(xI.shape, wI, zI_zM_g, *cp))))
        ctx_conv_func_back.save_for_backward(xR, xI, wR, wI, zR_zM, zI_zM, grad_output)
        ctx_conv_func_back.mark_non_differentiable(grad_wR, grad_wI, grad_bR, grad_bI)
        ctx_conv_func_back.cp = ctx.cp
        return grad_inp, grad_wR, grad_wI, grad_bR, grad_bI
    
    @staticmethod
    def backward(ctx, grad_inp_g, grad_wR_g, grad_wI_g, grad_bR_g, grad_bI_g):
        # only calculate gradient for grad_output_g, grad_wR, grad_wI
        grad_ctx = grad_output_g = grad_xR = grad_xI = grad_wR = grad_wI = grad_zR = grad_zI = grad_zM = None
        xR, xI, wR, wI, zR_zM, zI_zM, grad_output = ctx.saved_tensors
        xRg = xR.mul(grad_inp_g)
        xIg = xI.mul(grad_inp_g)
        zR_zM_g = zR_zM.mul(grad_output)
        zI_zM_g = zI_zM.mul(grad_output)
        cp = ctx.cp
        if ctx.needs_input_grad[1]:
            grad_output_g = (F.conv2d(xRg, wR, None, *cp).mul_(zI_zM).sub_(F.conv2d(xIg, wR, None, *cp).mul_(zR_zM)).sub_(F.conv2d(xRg, wI, None, *cp).mul_(zR_zM)).sub_(F.conv2d(xIg, wI, None, *cp).mul_(zI_zM)))
        if ctx.needs_input_grad[4]: # weightsR
            grad_wR = c2d_w(xRg, wR.shape, zI_zM_g, *cp).sub_(c2d_w(xIg, wR.shape, zR_zM_g, *cp)) #df(x)/dwI
        if ctx.needs_input_grad[5]: # weightsI
            grad_wI = (c2d_w(xRg, wI.shape, zR_zM_g, *cp).add_(c2d_w(xIg, wI.shape, zI_zM_g, *cp))).mul_(-1.) #-df(x)/dwR
        return grad_ctx, grad_output_g, grad_xR, grad_xI, grad_wR, grad_wI, grad_zR, grad_zI, grad_zM


class RealFC_Func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, w, b=None):
        ctx.save_for_backward(inp, w, b)
        output = inp.mm(w.t())
        if b is not None:
            output += b.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, w, b = ctx.saved_tensors
        grad_input = grad_w = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(w)
        if ctx.needs_input_grad[1]:
            grad_w = grad_output.t().mm(inp)
        if b is not None and ctx.needs_input_grad[2]:
            grad_b = grad_output.sum(0)
        return grad_input, grad_w, grad_b

class RealConv2d_Func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, w, b=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.save_for_backward(inp, w, b)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        out = F.conv2d(inp, w, b, stride, padding, dilation, groups)
        if b is not None:
            out = out + b.reshape(1, b.shape[0], 1, 1).expand_as(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_inp = grad_w = grad_b = None
        inp, w, b = ctx.saved_tensors
        cp = (ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        if ctx.needs_input_grad[0]:
            grad_inp = torch.nn.grad.conv2d_input(inp.shape, w, grad_output, *cp)
        if ctx.needs_input_grad[1]:
            grad_w = torch.nn.grad.conv2d_weight(inp, w.shape, grad_output, *cp)
        if b is not None and ctx.needs_input_grad[2]:
            # sum over minibatch and spatial dimensions
            grad_b = grad_output.sum(dim=(0, 2, 3)) * 2.
        return grad_inp, grad_w, grad_b, None, None, None, None

def test():

    class DummyCTX(object):
        def __init__(self, needs_input_grad, cp=(1, 1, 1, 1)):
            super(DummyCTX, self).__init__()
            self.needs_input_grad = needs_input_grad
            self.cp = cp

    from torch.autograd import gradcheck
    # sinfunc
    test = gradcheck(SinFunc.apply, (torch.randn(20, 20, dtype=torch.double, requires_grad=True),), eps=1e-6, atol=1e-4, raise_exception=True)
    print("SinFunc: ", test)

    input = (torch.randn(20,20,dtype=torch.double,requires_grad=True), torch.randn(30,20,dtype=torch.double,requires_grad=True), torch.randn(30,20,dtype=torch.double,requires_grad=True), \
        torch.randn(30, dtype=torch.double, requires_grad=True), torch.randn(30, dtype=torch.double, requires_grad=True))
    test = gradcheck(ComplexFC_Func.apply, input, eps=1e-6, atol=1e-4, raise_exception=False)
    print("ComplexFC Func Forward: ", test)

    input = (DummyCTX((True, True, True, True, True, True)), torch.randn(20, 30, dtype=torch.double, requires_grad=True), torch.randn(20, 20, dtype=torch.double, requires_grad=False), \
        torch.randn(20, 20, dtype=torch.double, requires_grad=False), torch.randn(30, 20, dtype=torch.double, requires_grad=True), torch.randn(30, 20, dtype=torch.double, requires_grad=True), \
            torch.randn(20, 30, dtype=torch.double, requires_grad=False), torch.randn(20, 30, dtype=torch.double, requires_grad=False), torch.randn(20, 30, dtype=torch.double, requires_grad=False))
    test = gradcheck(ComplexFC_Func_Backward.apply, input, eps=1e-6, atol=1e-4, raise_exception=True)
    print("ComplexFC Func Backward: ", test)

    input = (torch.randn(6, 3, 3, 3,dtype=torch.double,requires_grad=True), torch.randn(6, 3, 3, 3,dtype=torch.double,requires_grad=True), torch.randn(6, 3, 3, 3,dtype=torch.double,requires_grad=True), \
        torch.randn(6, dtype=torch.double, requires_grad=True), torch.randn(6, dtype=torch.double, requires_grad=True), 1, 1, 1, 1)
    test = gradcheck(ComplexConv2d_Func.apply, input, eps=1e-6, atol=1e-4, raise_exception=False)
    print("ComplexConv Func Forward: ", test)

    input = (DummyCTX((True, True, True, True, True, True)), torch.randn(6, 6, 3, 3,dtype=torch.double,requires_grad=True), torch.randn(6, 3, 3, 3,dtype=torch.double,requires_grad=False), \
        torch.randn(6, 3, 3, 3,dtype=torch.double,requires_grad=False), torch.randn(6, 3, 3, 3,dtype=torch.double,requires_grad=True), torch.randn(6, 3, 3, 3,dtype=torch.double,requires_grad=True), \
        torch.randn(6, 6, 3, 3,dtype=torch.double,requires_grad=False), torch.randn(6, 6, 3, 3,dtype=torch.double,requires_grad=False), torch.randn(6, 6, 3, 3,dtype=torch.double,requires_grad=False))
    test = gradcheck(ComplexConv2d_Func_Backward.apply, input, eps=1e-6, atol=1e-4, raise_exception=True)
    print("ComplexConv Func Backward", test)

    input = (torch.randn(20,20,dtype=torch.double,requires_grad=True), torch.randn(30,20,dtype=torch.double,requires_grad=True), torch.randn(30,dtype=torch.double,requires_grad=True))
    test = gradcheck(RealFC_Func.apply, input, eps=1e-6, atol=1e-4, raise_exception=False)
    print("RealFC Func: ", test)

    input = (torch.randn(6, 3, 3, 3, dtype=torch.double, requires_grad=True), torch.randn(6, 3, 3, 3, dtype=torch.double, requires_grad=True), torch.randn(6, dtype=torch.double, requires_grad=True),\
        1, 1, 1, 1)
    test = gradcheck(RealConv2d_Func.apply, input, eps=1e-6, atol=1e-4, raise_exception=False)
    print("RealConv Func: ", test)