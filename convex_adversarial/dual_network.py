import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .utils import Dense, DenseSequential
from .dual_inputs import select_input
from .dual_layers import select_layer

import numpy as np
import warnings


class DualNetwork(nn.Module):   
    def __init__(self, net, X, epsilon, 
                 l1_proj=None, l1_type='exact', bounded_input=False, 
                 data_parallel=True):
        """ 
        This class creates the dual network. 

        net : ReLU network
        X : minibatch of examples
        epsilon : size of l1 norm ball to be robust against adversarial examples
        alpha_grad : flag to propagate gradient through alpha
        scatter_grad : flag to propagate gradient through scatter operation
        l1 : size of l1 projection
        l1_eps : the bound is correct up to a 1/(1-l1_eps) factor
        m : number of probabilistic bounds to take the max over
        """
        super(DualNetwork, self).__init__()
        # need to change that if no batchnorm, can pass just a single example
        if not isinstance(net, (nn.Sequential, DenseSequential)): 
            raise ValueError("Network must be a nn.Sequential or DenseSequential module")
        with torch.no_grad(): 
            if any('BatchNorm2d' in str(l.__class__.__name__) for l in net): 
                zs = [X]
            else:
                zs = [X[:1]]
            nf = [zs[0].size()]
            for l in net: 
                if isinstance(l, Dense): 
                    zs.append(l(*zs))
                else:
                    zs.append(l(zs[-1]))
                nf.append(zs[-1].size())


        # Use the bounded boxes
        dual_net = [select_input(X, epsilon, l1_proj, l1_type, bounded_input)]

        for i,(in_f,out_f,layer) in enumerate(zip(nf[:-1], nf[1:], net)): 
            dual_layer = select_layer(layer, dual_net, X, l1_proj, l1_type, in_f, out_f, zs[i])

            # skip last layer
            if i < len(net)-1: 
                for l in dual_net: 
                    l.apply(dual_layer)
                dual_net.append(dual_layer)
            else: 
                self.last_layer = dual_layer

        self.dual_net = dual_net
        return 

    def forward(self, c):
        """ For the constructed given dual network, compute the objective for
        some given vector c """
        nu = [-c]
        nu.append(self.last_layer.T(*nu))
        for l in reversed(self.dual_net[1:]): 
            nu.append(l.T(*nu))
        dual_net = self.dual_net + [self.last_layer]
        
        return sum(l.objective(*nu[:min(len(dual_net)-i+1, len(dual_net))]) for
           i,l in enumerate(dual_net))

class DualNetBounds(DualNetwork): 
    def __init__(self, *args, **kwargs):
        warnings.warn("DualNetBounds is deprecated. Use the proper "
                      "PyTorch module DualNetwork instead. ")
        super(DualNetBounds, self).__init__(*args, **kwargs)

    def g(self, c):
        return self(c)

class RobustBounds(nn.Module): 
    def __init__(self, net, epsilon, **kwargs): 
        super(RobustBounds, self).__init__()
        self.net = net
        self.epsilon = epsilon
        self.kwargs = kwargs

    def forward(self, X, y, ind): 
        num_classes = self.net[-1].out_features
        dual = DualNetwork(self.net, X, self.epsilon, **self.kwargs)
        c = Variable(torch.eye(num_classes).type_as(X)[y].unsqueeze(1) 
                - torch.eye(num_classes).type_as(X)[ind, :].unsqueeze(0))
        if X.is_cuda:
            c = c.cuda()
        f = -dual(c)

        return f

## robust training for overall robustness 
def robust_loss(net, epsilon, X, y, size_average=True, **kwargs):
    targ_clas = range(net[-1].out_features)
    f = RobustBounds(net, epsilon, **kwargs)(X,y,targ_clas)
    err = (f.max(1)[1] != y)

    if size_average: 
        robust_err = err.sum().item()/X.size(0)
    ce_loss = nn.CrossEntropyLoss(reduction='elementwise_mean')(f, y)

    return ce_loss, robust_err

## robust training for cost-sensitive robustness 
def robust_loss_task_spec(net, epsilon, X, y, input_mat, mat_type, alpha=1.0, **kwargs):
    num_classes = net[-1].out_features
    # loss function for standard classification
    out = net(X)
    clas_err = (out.max(1)[1] != y).float().sum().item() / X.size(0)
    ce_loss = nn.CrossEntropyLoss(reduction='elementwise_mean')(out, y)

    # regularization term for cost-sensitive robustness
    cost_adv_exps = 0.0    
    num_exps = 0

    for k in range(num_classes):
        if np.all(input_mat[k, :] == 0):
            continue    
        else:
            targ_clas = np.nonzero(input_mat[k, :])[0]    # extract the corresponding output classes
            ind = (y == k).nonzero()   # extract the considered input example indices   

            if len(ind) != 0:
                ind = ind.squeeze(1)
                X_sub = X[ind, ...]
                y_sub = y[ind, ...]

                # robust score matrix
                f = RobustBounds(net, epsilon, **kwargs)(X_sub,y_sub,targ_clas)
                zero_col = torch.FloatTensor(np.zeros(len(ind), dtype=float)).cuda()
                weight_vec = torch.FloatTensor(input_mat[k, targ_clas]).repeat(len(ind),1).cuda() 

                # cost-weighted robust score matrix
                f_weighted = torch.cat((f + torch.log(weight_vec), zero_col.unsqueeze(1)), dim=1)
                target = torch.LongTensor(len(targ_clas)*np.ones(len(ind), dtype=float)).cuda()
                # aggregate the training loss function (including the robust regularizer)
                ce_loss = ce_loss + alpha*nn.CrossEntropyLoss(reduction='elementwise_mean')(f_weighted, target)

                zero_tensor = torch.FloatTensor(np.zeros(f.size())).cuda()
                err_mat = (f > zero_tensor).cpu().numpy()

                if mat_type == 'binary':    # same as the number of cost-sensitive adversarial exps
                    cost_adv_exps += err_mat.max(1).sum().item()               
                else:   # real-valued case
                    # use the total costs as the measure
                    cost_adv_exps += np.dot(np.sum(err_mat, axis=0), input_mat[k,targ_clas])
                num_exps += len(ind)
       
    return clas_err, ce_loss, cost_adv_exps, num_exps


## pairwise classification and robust error
def calc_err_clas_spec(net, epsilon, X, y, **kwargs):
    
    num_classes = net[-1].out_features
    targ_clas = range(num_classes)
    zero_mat = torch.FloatTensor(X.size(0), num_classes).zero_()
    # aggregate the class-specific classification and robust error counts
    clas_err_mat = torch.FloatTensor(num_classes+1, num_classes+1).zero_()
    robust_err_mat = torch.FloatTensor(num_classes+1, num_classes+1).zero_()
    # aggregate the number of examples for each class
    num_exps_vec = torch.FloatTensor(num_classes+1).zero_()
        
    if X.is_cuda:
        zero_mat = zero_mat.cuda()
        clas_err_mat = clas_err_mat.cuda()
        robust_err_mat = robust_err_mat.cuda()
        num_exps_vec = num_exps_vec.cuda()

    # compute the class-specific classification error matrix
    val, idx = torch.max(net(X), dim=1)
    for j in range(len(y)):
        row_ind = y[j]
        col_ind = idx[j].item()
        if row_ind != col_ind:
            clas_err_mat[row_ind, col_ind] += 1
            clas_err_mat[row_ind, num_classes] += 1
    clas_err_mat[num_classes, ] = torch.sum(clas_err_mat[:num_classes, ], dim=0)

    f = RobustBounds(net, epsilon, **kwargs)(X,y,targ_clas)
    # robust error counts for each example
    err_mat = (f > zero_mat).float()    # class-specific robust error counts
    err = (f.max(1)[1] != y).float()    # overall robust error counts

    # compute pairwise robust error matrix
    for i in range(num_classes):
        ind = (y == i).nonzero()    # indices of examples in class i
        if len(ind) != 0:
            ind = ind.squeeze(1)  
            robust_err_mat[i, :num_classes] += torch.sum(err_mat[ind, ].squeeze(1), dim=0)
            robust_err_mat[i, num_classes] += torch.sum(err[ind])
        num_exps_vec[i] += len(ind)

    # compute the weighted average for each target class 
    robust_err_mat[num_classes, ] = torch.sum(robust_err_mat[:num_classes, ], dim=0)
    num_exps_vec[num_classes] = torch.sum(num_exps_vec[:num_classes])

    return clas_err_mat, robust_err_mat, num_exps_vec

