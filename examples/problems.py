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
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from convex_adversarial import epsilon_from_model, DualNetBounds
from convex_adversarial import Dense, DenseSequential
import math


def model_wide(in_ch, out_width, k): 
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*k, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*k, 8*k, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*k*out_width*out_width,k*128),
        nn.ReLU(),
        nn.Linear(k*128, 10)
    )
    return model

def model_deep(in_ch, out_width, k, n1=8, n2=16, linear_size=100): 
    def group(inf, outf, N): 
        if N == 1: 
            conv = [nn.Conv2d(inf, outf, 4, stride=2, padding=1), 
                         nn.ReLU()]
        else: 
            conv = [nn.Conv2d(inf, outf, 3, stride=1, padding=1), 
                         nn.ReLU()]
            for _ in range(1,N-1):
                conv.append(nn.Conv2d(outf, outf, 3, stride=1, padding=1))
                conv.append(nn.ReLU())
            conv.append(nn.Conv2d(outf, outf, 4, stride=2, padding=1))
            conv.append(nn.ReLU())
        return conv

    conv1 = group(in_ch, n1, k)
    conv2 = group(n1, n2, k)


    model = nn.Sequential(
        *conv1, 
        *conv2,
        Flatten(),
        nn.Linear(n2*out_width*out_width,linear_size),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def mnist_loaders(batch_size, ratio=0, seed=None, is_shuffle=False): 
    mnist_train_total = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())

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

def mnist_model_wide(k): 
    return model_wide(1, 7, k)

def mnist_model_deep(k): 
    return model_deep(1, 7, k)

def mnist_model_large(): 
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*7*7,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model

def replace_10_with_0(y): 
    return y % 10

def cifar_loaders(batch_size, ratio=0, seed=None, is_shuffle=False): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train_total = datasets.CIFAR10('./data', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./data', train=False, 
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

def cifar_model_resnet(N = 5, factor=10): 
    def  block(in_filters, out_filters, k, downsample): 
        if not downsample: 
            k_first = 3
            skip_stride = 1
            k_skip = 1
        else: 
            k_first = 4
            skip_stride = 2
            k_skip = 2
        return [
            Dense(nn.Conv2d(in_filters, out_filters, k_first, stride=skip_stride, padding=1)), 
            nn.ReLU(), 
            Dense(nn.Conv2d(in_filters, out_filters, k_skip, stride=skip_stride, padding=0), 
                  None, 
                  nn.Conv2d(out_filters, out_filters, k, stride=1, padding=1)), 
            nn.ReLU()
        ]
    conv1 = [nn.Conv2d(3,16,3,stride=1,padding=1), nn.ReLU()]
    conv2 = block(16,16*factor,3, False)
    for _ in range(N): 
        conv2.extend(block(16*factor,16*factor,3, False))
    conv3 = block(16*factor,32*factor,3, True)
    for _ in range(N-1): 
        conv3.extend(block(32*factor,32*factor,3, False))
    conv4 = block(32*factor,64*factor,3, True)
    for _ in range(N-1): 
        conv4.extend(block(64*factor,64*factor,3, False))
    layers = (
        conv1 + 
        conv2 + 
        conv3 + 
        conv4 +
        [Flatten(),
        nn.Linear(64*factor*8*8,1000), 
        nn.ReLU(), 
        nn.Linear(1000, 10)]
        )
    model = DenseSequential(
        *layers
    )
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None: 
                m.bias.data.zero_()
    return model

def argparser(prefix=None, method=None, batch_size=50, epochs=60, 
              verbose=200, lr=1e-3, thres=0.02, ratio=0.2,
              seed=0, epsilon=0.1, starting_epsilon=None, 
              l1_proj=None, delta=None, m=1, l1_eps=None, 
              l1_train='exact', l1_test='exact', 
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
    parser.add_argument('--delta', type=float, default=delta)
    parser.add_argument('--m', type=int, default=m)
    parser.add_argument('--l1_train', default=l1_train)
    parser.add_argument('--l1_test', default=l1_test)
    parser.add_argument('--l1_eps', type=float, default=l1_eps)

    # model arguments
    parser.add_argument('--model', default=None)
    parser.add_argument('--model_factor', type=int, default=8)
    parser.add_argument('--cascade', type=int, default=1)
    parser.add_argument('--method', type=str, default=method)
    parser.add_argument('--resnet_N', type=int, default=1)
    parser.add_argument('--resnet_factor', type=int, default=1)

    # task-specific arguments
    parser.add_argument('--type', default=None)
    parser.add_argument('--category', default=None)
    parser.add_argument('--tuning', default=None)

    # other arguments
    parser.add_argument('--prefix', type=str, default=prefix)
    parser.add_argument('--ratio', type=float, default=ratio)
    parser.add_argument('--thres', type=float, default=thres)
    parser.add_argument('--load')
    parser.add_argument('--real_time', action='store_true')
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

        # if args.epsilon is not None:
        #     args.prefix += 'epsilon_'+args.epsilon+'/'

        banned = ['proctitle', 'verbose', 'prefix', 'opt', 'real_time',
                  'batch_size', 'lr', 'ratio', 'thres', 'momentum', 'weight_decay',
                  'l1_test', 'l1_train', 'type', 'category', 'tuning',
                  'seed', 'method', 'model', 'cuda_ids', 'load']
                
        if args.method == 'baseline':
            banned += ['starting_epsilon', 'schedule_length', 
                       'l1_test', 'l1_train', 'm', 'l1_proj', 'thres']

        if args.method == 'task_spec_robust':
            if args.type is not None:
                args.proctitle += args.type+'/'
            if args.category is not None:
                args.proctitle += args.category+'/'
            if args.tuning == 'coarse':
                banned += ['thres']

        # # if not using adam, ignore momentum and weight decay
        # if args.opt == 'adam': 
        #     banned += ['momentum', 'weight_decay']

        if args.m == 1: 
            banned += ['m']
        if args.cascade == 1: 
            banned += ['cascade']

        # if not using a model that uses model_factor, 
        # ignore model_factor
        if args.model not in ['wide', 'deep']: 
            banned += ['model_factor']

        if args.model != 'resnet': 
            banned += ['resnet_N', 'resnet_factor']

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

# parameters l1_eps, m, delta are used for geometric estimation for testing
def args2kwargs(model, args, X=None): 
    if args.l1_proj is not None: 
        if not args.l1_eps:
            if args.delta: 
                args.l1_eps = epsilon_from_model(model, Variable(X.cuda()), args.l1_proj,
                                            args.delta, args.m)
                print('''
        With probability {} and projection into {} dimensions and a max
        over {} estimates, we have epsilon={}'''.format(args.delta, args.l1_proj,
                                                        args.m, args.l1_eps))
            else: 
                args.l1_eps = 0
                print('No epsilon or \delta specified, using epsilon=0.')
        else:
            print('Specified l1_epsilon={}'.format(args.l1_eps))
        kwargs = {
            'l1_proj' : args.l1_proj, 
            # 'l1_eps' : args.l1_eps, 
            # 'm' : args.m
        }
    else:
        kwargs = {
        }
    return kwargs