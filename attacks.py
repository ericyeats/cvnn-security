import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms


# estimate the gradient of the confidence score w.r.t. the input data
def nes_gradient_est(model, device, data, lbl, n_query=100, sigma=0.001):
    model.eval()
    g = torch.zeros_like(data)
    inds = range(lbl.shape[0]), lbl
    with torch.no_grad():
        for q in range(n_query):
            n = torch.randn_like(g)
            # accumulate prediction score * antithetic samples
            scores = model(data + sigma*n)
            g.add_(n * scores[inds].view(-1, 1, 1, 1))
            scores = model(data - sigma*n)
            g.sub_(n * scores[inds].view(-1, 1, 1, 1))
    return ((-1./(2*n_query*sigma))*g).detach()

# Compute the gradient of the loss w.r.t. the input data
def gradient_wrt_data(model,device,data,lbl):
    dat = data.clone().detach()
    dat.requires_grad = True
    out = model(dat)
    loss = F.cross_entropy(out,lbl)
    model.zero_grad()
    loss.backward()
    data_grad = dat.grad.data
    return data_grad.data.detach()

def PGD_attack(model, device, dat, lbl, eps, alpha, iters, rand_start, norm, inorm, l2=False, nes=False):
    # IMPORTANT - expect data to be normalized
    # - dat and lbl are tensors
    # - eps and alpha are floats
    # - iters is an integer
    # - rand_start is a bool
    # x0 is random jump around dat
    dat = inorm(dat)
    dat0 = dat.clone().detach()
    if rand_start and not l2:
        dat = dat + torch.empty_like(dat).uniform_(-eps, eps) # make random jump with uniform distribution in eps ball
        dat = torch.clamp(dat, 0, 1) # must stay in image bounds; no normalization
    dat = norm(dat)
    for i in range(iters):
        grad = gradient_wrt_data(model, device, dat, lbl) if not nes else nes_gradient_est(model, device, dat, lbl)
        grad_n = grad.view(-1, grad.numel()//grad.shape[0]).norm(dim=1, p=2).view(-1, 1, 1, 1) if l2 else torch.ones(1).to(grad.device)
        grad = torch.where(grad_n.expand_as(grad) > 0.0, grad, torch.randn_like(grad))
        direct = grad/grad_n if l2 else grad.sign()
        dat = inorm(dat)
        if l2:
            direct = inorm.scale_inorm(direct)
        dat = dat + alpha * direct
        dat = torch.clamp(dat, 0, 1)
        if l2 and eps>0:
            diff = dat - dat0
            dist = diff.view(-1, diff.numel()//diff.shape[0]).norm(dim=1, p=2).view(-1, 1, 1, 1)
            proj = dat0 + (eps*(diff/dist)) # project onto l2 ball
            dat = torch.where(dist.expand_as(dat) > eps, proj, dat)
        else:
            dat = (dat0 + torch.clamp(dat - dat0, -eps, eps)).detach()
        dat = norm(dat)
    return dat

def FGSM_attack(model, device, dat, lbl, eps, norm, inorm):
    # - Dat and lbl are tensors
    # - eps is a float
    return PGD_attack(model, device, dat, lbl, eps, eps, 1, False, norm, inorm)
