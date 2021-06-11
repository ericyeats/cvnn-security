import torchvision.datasets as dsets
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

class NumpyData(torch.utils.data.Dataset):
    """
    Inherits from torch.utils.data.Dataset
    Reads a numpy
    """
    def __init__(self, imagespath, labelspath):
        super(NumpyData, self).__init__()
        self.imdata = np.load(imagespath, allow_pickle=True)
        self.labdata = np.load(labelspath, allow_pickle=True)

    def __getitem__(self, index):
        return self.imdata[index], self.labdata[index]

    def __len__(self):
        return self.imdata.shape[0]

def eval_net(net, testloader, device, norm, inorm, crit=torch.nn.CrossEntropyLoss()):
    correct = 0
    total = 0
    loss = 0
    net.eval()
    n_batch = 0
    with torch.no_grad():
        for data in testloader:
            n_batch += 1
            images, labels = norm(data[0].to(device)), data[1].to(device)
            outputs = net(images)
            loss += crit(outputs, labels)
            images = inorm(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total, loss, n_batch

def get_tforms(task, use_noise=False):
    train_tforms, test_tforms = None, None
    if task == 'svhn':
        train_tforms = [
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            Luminance(),
            ]
        test_tforms = [
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            Luminance()]
    elif task == 'fmnist':
        train_tforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=1, padding_mode='edge'),
            transforms.ToTensor(),
            ]
        test_tforms = [
            transforms.ToTensor()]
    elif task == 'mnist':
        train_tforms = [
            transforms.ToTensor()]
        test_tforms = [
            transforms.ToTensor()]
    elif task == 'cifar':
        train_tforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='edge'),
            transforms.ToTensor()]
        test_tforms = [
            transforms.ToTensor()]
    if use_noise:
        train_tforms.append(GaussianNoise(sigma=0.05))
        train_tforms.append(Clamp())
    return train_tforms, test_tforms


def get_norm_inorm(task, useComp=False):
    norm, inorm = None, None
    if task == 'svhn':
        norm = inorm = Identity()
    elif task == 'fmnist':
        norm = inorm = Identity()
    elif task == 'mnist':
        norm = inorm = Identity()
    elif task == 'cifar':
        if useComp:
            norm = torch.nn.Identity()
            inorm = torch.nn.Identity()
        else:
            norm = cifar_norm
            inorm = cifar_inorm
    return norm, inorm


def get_train_params(task):
    lr, batch_size, num_epochs, milestones = None, None, None, None
    if task == 'mnist' or task == 'fmnist' or task == 'svhn':
        lr = 0.005
        batch_size = 64
        num_epochs = 30
        milestones = [20]
    elif task == 'cifar':
        lr = 0.01
        batch_size = 128
        num_epochs = 80
        milestones = [int(num_epochs*0.5), int(num_epochs*0.75), int(num_epochs*0.9)]
    return lr, batch_size, num_epochs, milestones

def get_train_test_set(task, train_transform, test_transform):
    trainset, testset = None, None
    rootdir = '../torch_datasets/'
    if task == 'cifar':
        tset = dsets.CIFAR10
        trainset = tset(root=rootdir, train=True,
                download=True, transform=train_transform)
        testset = tset(root=rootdir, train=False,
                download=True, transform=test_transform)
    elif task == 'fmnist':
        tset = dsets.FashionMNIST
        trainset = tset(root=rootdir, train=True,
                download=True, transform=train_transform)
        testset = tset(root=rootdir, train=False,
                download=True, transform=test_transform)
    elif task == 'mnist':
        tset = dsets.MNIST
        trainset = tset(root=rootdir, train=True,
                download=True, transform=train_transform)
        testset = tset(root=rootdir, train=False,
                download=True, transform=test_transform)
    elif task == 'svhn':
        tset = dsets.SVHN
        trainset = tset(root=rootdir, split='train',
                download=True, transform=train_transform)
        testset = tset(root=rootdir, split='test',
                download=True, transform=test_transform)
    return trainset, testset

def get_epsilons(task, l2=False):
    epsilons = None
    if task == 'cifar':
        epsilons = [0.0, 1./255, 2./255, 3./255, 4./255, 5./255, 6./255, 7./255, 8./255]
    elif task == 'fmnist' or task == 'mnist' or task == 'svhn':
        if l2:
            epsilons = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        else:
            epsilons = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.3]
    return epsilons

def num_params(net):
    return sum(p.numel() for p in net.parameters())

# thanks Joel Simon at https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/2

class UnNormalize(object):
    def __init__(self, mean, std):
        super(UnNormalize, self).__init__()
        self.mean = torch.tensor(mean).reshape((len(mean), 1, 1))
        self.std = torch.tensor(std).reshape((len(std), 1, 1))

    def __call__(self, tensor):
        return self.scale_inorm(tensor).add(self.mean.to(tensor.device))

    def scale_inorm(self, tensor):
        return tensor.mul(self.std.to(tensor.device))

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).reshape((len(mean), 1, 1))
        self.std = torch.tensor(std).reshape((len(std), 1, 1))

    def __call__(self, tensor):
        return tensor.sub(self.mean.to(tensor.device)).div(self.std.to(tensor.device))

class Identity(object):
    def __init__(self):
        super(Identity, self).__init__()
    
    def __call__(self, tensor):
        return tensor
    
    def scale_inorm(self, tensor):
        return tensor

cifar_norm = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
cifar_inorm = UnNormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

class Clamp(object):

    def __init__(self, _min=0, _max=1):
        super(Clamp, self).__init__()
        self._min = _min
        self._max = _max
    
    def __call__(self, inp):
        return torch.clamp(inp, self._min, self._max)

class GaussianNoise(object):

    def __init__(self, sigma=0.1):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
    
    def __call__(self, inp):
        return inp + torch.randn_like(inp) * self.sigma
        
class Luminance(object):

    def __init__(self):
        super(Luminance, self).__init__()
        self.l = torch.tensor([[0.2989, 0.5870, 0.1140]])

    def __call__(self, inp):
        sh = inp.shape
        inp = self.l.matmul(inp.view(3, sh[1]*sh[2])).view(1, sh[1], sh[2])
        inp = (inp - inp.min()) / (inp.max() - inp.min())
        return inp
