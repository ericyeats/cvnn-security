import numpy as np
import torch
import pytorch_helpers as phelps
from attacks import PGD_attack

import time


check_input_grad = True

# train a network on a specified device
# capable of gradient regularization or adversarial training with PGD
def train_network(net, device, lr, num_epochs, milestones, beta, trainloader, testloader, norm, inorm, advTrain, advT_eps, advT_steps, advT_jump):
    net.to(device)
    print("Num Params: {}".format(phelps.num_params(net)))

    # loss, val_loss, smooth_loss, train_acc, val_acc, batch_sim
    stats = np.zeros((6, num_epochs))
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.875, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)

    start_time = time.time()
    net.train()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        correct = 0
        total = 0
        sim_track = 0.0

        running_loss = 0.0
        running_smooth_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            print("\r{:2.1f}%".format((i+1)/len(trainloader) * 100.) , end="")
            param_names = []
            standard_grad = []
            grad_inp_grad = []

            # get the inputs
            inputs, labels = torch.autograd.Variable(data[0]).to(device) ,data[1].to(device)
            total += inputs.shape[0]
            # manually apply norm
            inputs = norm(inputs)
            inputs.requires_grad = True

            # zero the parameter gradients
            optimizer.zero_grad()
            if advTrain:
                inputs = PGD_attack(net, device, inputs, labels, advT_eps, 1.85*(advT_eps)/advT_steps, advT_steps, advT_jump, norm, inorm)
                optimizer.zero_grad()
            net.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            correct += (outputs.max(dim=1)[1] == labels).sum().item()
            losses = criterion(outputs, labels)
            loss = losses.mean()

            loss.backward(create_graph=beta>0.0)

            if check_input_grad:
                for name, param in net.named_parameters():
                    if param.requires_grad:
                        param_names.append(name)
                        standard_grad.append(param.grad.view(param.grad.numel()).clone().detach())

            if advTrain:
                inputs = inorm(inputs) # clear norm state

            if beta > 0.0: # don't want to call .backward() if not connected to graph
                smooth_loss = beta * (torch.norm(inputs.grad.view(-1, inputs.numel()//inputs.shape[0]), p=1, dim=1)**2).mean()
                smooth_loss.backward()
                inputs.grad=None
                running_smooth_loss += smooth_loss.item() / (beta)

            optimizer.step()

            if check_input_grad:
                for name, param in net.named_parameters():
                    if param.requires_grad:
                        grad_inp_grad.append(param.grad.view(param.grad.numel()).clone().detach())
                dots = sum([g.dot(s) for g, s in zip(grad_inp_grad, standard_grad)]).item()
                g_len = sum([g.dot(g) for g in grad_inp_grad]).sqrt().item()
                s_len = sum([s.dot(s) for s in standard_grad]).sqrt().item()
                denom = g_len*s_len
                sim_track += dots/denom if denom != 0 else 1.0
                    

            # print statistics
            running_loss += loss.item()
            del loss
            del inputs
            del labels
        # run validation
        val_correct, val_total, val_loss, val_batch = phelps.eval_net(net, testloader, device, norm, inorm)
        net.train()
        net.rgrad = True
        stats[0][epoch] = running_loss / len(trainloader)
        stats[1][epoch] = val_loss/val_batch
        stats[2][epoch] = running_smooth_loss / len(trainloader)
        stats[3][epoch] = correct/total
        stats[4][epoch] = val_correct/val_total
        stats[5][epoch] = sim_track/len(trainloader)
        print('\r[%d] loss: %.7f val_loss: %.7f smooth loss: %.5f train_acc: %.3f val_acc: %.3f batch_sim: %.3f' %
                (epoch + 1, stats[0][epoch], stats[1][epoch], stats[2][epoch], stats[3][epoch], stats[4][epoch], stats[5][epoch]))
        running_loss = 0.0
        running_smooth_loss = 0.0
        scheduler.step()
    end_time = time.time()
    
    correct, total, _, _ = phelps.eval_net(net, testloader, device, norm, inorm)

    print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
    print('Finished Training')
    print('Training Time: {:.2f}'.format(end_time - start_time))
    return stats





