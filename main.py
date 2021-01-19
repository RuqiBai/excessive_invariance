from models.RevNet import Excessive
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch.nn import Parameter
from models.model_utils import get_all_params

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 30):
        optim_factor = 1
    elif(epoch > 60):
        optim_factor = 2
    elif(epoch > 90):
        optim_factor = 3
    return init*math.pow(0.2, optim_factor)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
def train(model, trainloader, trainset, epoch, num_epochs, batch_size, lr, use_cuda, in_shape):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.Adamax(model.parameters(), lr=learning_rate(lr, epoch), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print('|  Number of Trainable Parameters: ' + str(params))
    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, learning_rate(lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        out, out_bij,_, _ = model(inputs)               # Forward Propagation
        loss = criterion(out, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        try:
            loss.data[0]
        except IndexError:
            loss.data = torch.reshape(loss.data, (1,))
        train_loss += loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                         % (epoch, num_epochs, batch_idx+1,
                            (len(trainset)//batch_size)+1, loss.data[0], 100.*correct/total))
        sys.stdout.flush()


def test(model, testloader, testset, epoch, use_cuda, best_acc, dataset, fname):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out, out_bij, _, _ = model(inputs)
        loss = criterion(out, targets)

        try:
            loss.data[0]
        except IndexError:
            loss.data = torch.reshape(loss.data, (1,))
        test_loss += loss.data[0]
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*correct/total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.data[0], acc))

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
        state = {
                'model': model if use_cuda else model,
                'acc': acc,
                'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/'+dataset+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+fname+'.t7')
        best_acc = acc
    return best_acc


def main():
    epochs = 1
    batch = 64
    lr = 0.001
    elapsed_time = 0
    best_acc = 0.
    in_shape = [1,32,32]
    dataset = "MNIST"
    fname = "excessive"
    model = Excessive(nBlocks=[16, 16, 16], nStrides=[1, 1, 1], nClasses=10, init_ds=2,
                      dropout_rate=0., affineBN=True, in_shape=(1, 32, 32), mult=1)

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    test_train = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch)


    use_cuda = True
    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=(0,))  # range(torch.cuda.device_count()))
        cudnn.benchmark = True

        for epoch in range(1, 1 + epochs):
            start_time = time.time()

            train(model, trainloader, trainset, epoch, epochs, batch, lr, use_cuda, in_shape)
            best_acc = test(model, testloader, testset, epoch, use_cuda, best_acc, dataset, fname)

            epoch_time = time.time() - start_time
            elapsed_time += epoch_time
            print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

        print('Testing model')
        print('* Test results : Acc@1 = %.2f%%' % (best_acc))


if __name__ == "__main__":
    main()
