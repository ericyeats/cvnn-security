import argparse
import architectures as arch
import pytorch_helpers as phelps
from train_utils import train_network
from attack_utils import adv_test
import torch
import torchvision.transforms as transforms
import numpy as np

# General Arguments

parser = argparse.ArgumentParser(description="CVNN & Gradient Regularization")
parser.add_argument("task", help="\"train\", \"attack\", or \"eval\"", choices=["train", "attack", "eval"])
parser.add_argument("arch", help="select a network architecture", choices=arch.arch_names)
parser.add_argument("--use_complex", help="flag to use the complex version", action="store_true")
parser.add_argument("--net_load_path", help="path from which to load a .pt file of the specified architecture")

# arguments for TRAINING ONLY
parser.add_argument("--net_save_path", help="path to save a .pt file after training", default='./models/default.pt')
parser.add_argument("--beta", help='beta controls the strength of gradient regularization', type=float, default=0.0)
parser.add_argument("--use_noise", help='use gaussian noise N~(mu=0.0, sigma=0.05) during training', action="store_true")
parser.add_argument("--advtrain", help='adversarially train this network', action="store_true")

# arguments for TRAINING OR ATTACK
parser.add_argument("--dataset", help="dataset used to train or create an attack", choices=["mnist", "fmnist", "svhn", "cifar"], default="mnist")
parser.add_argument("--attack_eps", help="L-infinity bound for adversarial attack", type=float, default=0.1)
parser.add_argument("--attack_steps", help="number of steps for the attack", type=int, default=4)
parser.add_argument("--attack_jump", help="apply clipped random jump at the start of the attack", action="store_true")

# arguments for ATTACK
parser.add_argument("--nes", help="Employ a gradient-free NES attack against the network", action="store_true")
parser.add_argument("--save_examples", help="save the generated examples", action="store_true")
parser.add_argument("--examples_save_path", help="path to which to save crafted adversarial examples", default="./adv_examples/last_examples.npy")

# arguments for EVAL
parser.add_argument("--eval_adv_examples", help="evaluate from previously generated adv examples (rather than a clean dataset)", action="store_true")
parser.add_argument("--examples_load_path", help="path from which to load examples to evaluate the network", default="./adv_examples/last_examples.npy")

args = parser.parse_args()

# begin setting up the environment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = ((arch.arch_dict[args.arch])(args.use_complex)).to(device) # fetch the network architecture

if args.net_load_path:
    net.load_state_dict(torch.load(args.net_load_path))
    print("Loaded from: ", args.net_load_path)
else:
    print("No load path supplied! Starting from random init")


# Set up training, eval, or attack resources. May not need some of these.
lr, batch_size, num_epochs, milestones = phelps.get_train_params(args.dataset)

train_tforms, test_tforms = phelps.get_tforms(args.dataset, use_noise=args.use_noise)
norm, inorm = phelps.get_norm_inorm(args.dataset, useComp=args.use_complex)
train_transform = transforms.Compose(train_tforms)
test_transform = transforms.Compose(test_tforms)
trainset, testset = phelps.get_train_test_set(args.dataset, train_transform, test_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)


# TRAINING
if args.task == "train":

    if args.net_save_path:
        print("Will save as: ", args.net_save_path)

    print("LR: ", lr, "\t epochs: ", num_epochs, "\t beta: ", args.beta)

    stats = train_network(net, device, lr, num_epochs, milestones, args.beta, trainloader, \
        testloader, norm, inorm, args.advtrain, args.attack_eps, args.attack_steps, args.attack_jump)

    if args.net_save_path:
        torch.save(net.state_dict(), args.net_save_path)
        print("saved model at: ", args.net_save_path)
        np.save(args.net_save_path + "_train_stats.npy", stats, allow_pickle=True)
    

# ATTACK
if args.task == "attack":
    
    final_acc, adv_examples_npy, labels_npy = adv_test(net, device, args.dataset, testloader, \
        1000, args.attack_eps, args.attack_steps, args.attack_jump, norm, inorm, args.nes)

    if args.save_examples:
        np.save(args.examples_save_path, adv_examples_npy, allow_pickle=True)
        np.save(args.examples_save_path + "_labels.npy", labels_npy, allow_pickle=True)

# EVAL
if args.task == "eval":

    # if use prev. gen adversarial examples, must override testloader
    if args.eval_adv_examples and args.examples_load_path:
        print("Using examples from : ", args.examples_load_path)
        testset = phelps.NumpyData(args.examples_load_path, args.examples_load_path + "_labels.npy")
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    correct, total, _, _ = phelps.eval_net(net, testloader, device, norm, inorm)
    print("Accuracy: ", 100*correct/total, "%")
