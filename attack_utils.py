import numpy as np
from attacks import PGD_attack

def adv_test(net, device, task, test_loader, n_ims, epsilon, iterations, useJump, norm, inorm, NES):
    batch_size=100
    net.eval()
    net.to(device)
    shp = (n_ims, 1, 28, 28) if (task == "fmnist" or task == "mnist" or task == 'svhn') else (n_ims, 3, 32, 32)
    # Accuracy counter
    correct = 0
    adv_examples = np.empty(shp, dtype=np.float32)
    labels = np.empty((n_ims,), dtype=np.int8)
    # Loop over all examples in test set
    for b_id, (data, target) in enumerate(test_loader):

        if b_id * batch_size >= n_ims:
            break

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        data = norm(data) # data is expected to be normed already

        data = PGD_attack(net, device, data, target, epsilon, 1.85*epsilon/iterations, iterations, useJump, norm, inorm, l2=False, nes=NES)

        # Re-classify the perturbed image
        output = net(data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1].squeeze() # get the index of the max log-probability
        cor_batch = (final_pred == target).sum().item()
        correct += cor_batch
            
        # Special case for saving 0 epsilon examples
        data = inorm(data)
        adv_ex = data.squeeze().detach().cpu().numpy()
        inds = range(b_id*batch_size, (b_id*batch_size) + target.shape[0])
        adv_examples[inds] = np.reshape(adv_ex, (target.shape[0],) + shp[1:])
        labels[inds] = target.detach().cpu().numpy()
        print("\rEps: {} Adv Image: {}\t".format(epsilon, b_id*batch_size + target.shape[0]), end="")
    print()
    # Calculate final accuracy for this epsilon
    final_acc = correct/float(n_ims)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, n_ims, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples, labels