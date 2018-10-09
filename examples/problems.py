import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

import numpy as np
import torch.utils.data as td
import argparse
import os
import math

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def mnist_loaders(batch_size, path, ratio=0, seed=None, is_shuffle=False): 
    mnist_train_total = datasets.MNIST(path, train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(path, train=False, download=True, transform=transforms.ToTensor())

    if ratio == 0:
        train_loader = td.DataLoader(mnist_train_total, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = td.DataLoader(mnist_test, batch_size=batch_size, shuffle=is_shuffle, pin_memory=True)
        return train_loader, test_loader
    
    else:        
        # train-validation split based on the value of ratio
        num_train = int((1-ratio)*len(mnist_train_total))
        num_valid = len(mnist_train_total) - num_train
        
        torch.manual_seed(seed)
        mnist_train, mnist_valid = td.random_split(mnist_train_total, [num_train, num_valid])    

        train_loader = td.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
        valid_loader = td.DataLoader(mnist_valid, batch_size=batch_size, shuffle=is_shuffle, pin_memory=True)
        test_loader = td.DataLoader(mnist_test, batch_size=batch_size, shuffle=is_shuffle, pin_memory=True)
        return train_loader, valid_loader, test_loader

def mnist_model(): 
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def cifar_loaders(batch_size, ratio=0, seed=None, is_shuffle=False, path='./data'): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train_total = datasets.CIFAR10(path, train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10(path, train=False, 
                    transform=transforms.Compose([transforms.ToTensor(), normalize]))

    if ratio == 0:
        train_loader = td.DataLoader(train_total, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = td.DataLoader(test, batch_size=batch_size, shuffle=is_shuffle, pin_memory=True)
        return train_loader, test_loader
    
    else:     
        # train-validation split based on the value of ratio
        num_train = int((1-ratio)*len(train_total))
        num_valid = len(train_total) - num_train
        
        torch.manual_seed(seed)
        train, valid = td.random_split(train_total, [num_train, num_valid])    

        train_loader = td.DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
        valid_loader = td.DataLoader(valid, batch_size=batch_size, shuffle=is_shuffle, pin_memory=True)
        test_loader = td.DataLoader(test, batch_size=batch_size, shuffle=is_shuffle, pin_memory=True)
        return train_loader, valid_loader, test_loader

def cifar_model(): 
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model

def cifar_model_large(): 
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model

# define the argparser here for simplicity
def argparser(prefix=None, method=None, batch_size=50, epochs=60, 
              verbose=200, lr=1e-3, thres=0.02, ratio=0.2,
              seed=0, epsilon=0.1, starting_epsilon=None, 
              l1_proj=None, l1_train='exact', l1_test='exact', 
              opt='sgd', momentum=0.9, weight_decay=5e-4): 

    parser = argparse.ArgumentParser()

    # optimizer settings
    parser.add_argument('--opt', default=opt)
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)

    # epsilon settings
    parser.add_argument("--epsilon", type=float, default=epsilon)
    parser.add_argument("--starting_epsilon", type=float, default=starting_epsilon)
    parser.add_argument('--schedule_length', type=int, default=20)

    # projection settings
    parser.add_argument('--l1_proj', type=int, default=l1_proj)
    parser.add_argument('--l1_train', default=l1_train)
    parser.add_argument('--l1_test', default=l1_test)

    # model arguments
    parser.add_argument('--model', default=None)
    parser.add_argument('--method', type=str, default=method)

    # task-specific arguments
    parser.add_argument('--type', default=None)
    parser.add_argument('--category', default=None)
    parser.add_argument('--tuning', default=None)

    # other arguments
    parser.add_argument('--prefix', type=str, default=prefix)
    parser.add_argument('--ratio', type=float, default=ratio)
    parser.add_argument('--thres', type=float, default=thres)
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--verbose', type=int, default=verbose)
    parser.add_argument('--cuda_ids', default=None)
    parser.add_argument('--proctitle', type=str, default="")

    args = parser.parse_args()
    if args.starting_epsilon is None:
        args.starting_epsilon = args.epsilon

    if args.prefix:
        args.proctitle += args.prefix + '/'
        if args.model is not None: 
            args.proctitle += args.model + '/'

        if args.method is not None: 
            args.proctitle += args.method+'/'

        banned = ['proctitle', 'prefix', 'model', 'method', 'opt',
                  'momentum', 'weight_decay', 'batch_size', 'lr',
                  'l1_test', 'l1_train', 'type', 'category', 'tuning',
                  'ratio', 'thres', 'seed', 'verbose', 'cuda_ids']
                
        if args.method == 'baseline':
            banned += ['starting_epsilon', 'schedule_length', 'l1_proj']

        if args.method == 'task_spec_robust':
            if args.type is not None:
                args.proctitle += args.type+'/'
            if args.category is not None:
                args.proctitle += args.category+'/'
            if args.tuning == 'coarse':
                banned += ['thres']

        index = 0
        for arg in sorted(vars(args)): 
            if arg not in banned and getattr(args,arg) is not None: 
                if index == 0:
                    args.proctitle += arg + '_' + str(getattr(args, arg))
                else:
                    args.proctitle += '_' + arg + '_' + str(getattr(args, arg))
                index += 1

        if args.schedule_length >= args.epochs: 
            raise ValueError('Schedule length for epsilon ({}) is greater than '
                             'number of epochs ({})'.format(args.schedule_length, args.epochs))
    else: 
        args.proctitle = 'temporary'

    if args.cuda_ids is not None: 
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids
    return args

def args2kwargs(args): 
    if args.l1_proj is not None: 
        kwargs = {'l1_proj' : args.l1_proj}
    else:
        kwargs = {}
    return kwargs