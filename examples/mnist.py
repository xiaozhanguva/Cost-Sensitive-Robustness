import waitGPU
waitGPU.wait(utilization=20, available_memory=10000, interval=10)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import problems as pblm
from trainer import *
import setproctitle

def select_model(m): 
    if m == 'large': 
        model = pblm.mnist_model_large().cuda()
        _, test_loader = pblm.mnist_loaders(8)
    elif m == 'wide': 
        print("Using wide model with model_factor={}".format(args.model_factor))
        _, test_loader = pblm.mnist_loaders(64//args.model_factor)
        model = pblm.mnist_model_wide(args.model_factor).cuda()
    elif m == 'deep': 
        print("Using deep model with model_factor={}".format(args.model_factor))
        _, test_loader = pblm.mnist_loaders(64//(2**args.model_factor))
        model = pblm.mnist_model_deep(args.model_factor).cuda()
    else: 
        model = pblm.mnist_model().cuda() 
    return model

if __name__ == "__main__": 
    args = pblm.argparser(prefix='mnist', opt='adam', starting_epsilon=0.05, epsilon=0.2, thres=0.04)
    setproctitle.setproctitle('python')
    print("saving file to {}".format(args.proctitle))

    if args.method == 'overall_robust':
        print("threshold for classification error: {:.1%}".format(args.thres))

    saved_filepath = ('../saved_log/'+args.proctitle)
    model_filepath = os.path.dirname('../models/'+args.proctitle)
    if not os.path.exists(saved_filepath):
        os.makedirs(saved_filepath)
    if not os.path.exists(model_filepath):
        os.makedirs(model_filepath)
    model_path = ('../models/'+args.proctitle+'.pth')

    train_log = open(saved_filepath + '/train_log.txt', "w")
    train_res = open(saved_filepath + '/train_res.txt', "w")
    valid_res = open(saved_filepath + '/valid_res.txt', "w")
    best_res = open(saved_filepath + "/best_res.txt", "w")

    # train-validation split
    train_loader, valid_loader, test_loader = pblm.mnist_loaders(args.batch_size, args.ratio, args.seed)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    model = select_model(args.model)

    for X,y in train_loader: 
        kwargs = pblm.args2kwargs(model, args, X=Variable(X.cuda()))
        break

    if args.opt == 'adam': 
        opt = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd': 
        opt = optim.SGD(model.parameters(), lr=args.lr, 
                        momentum=args.momentum, weight_decay=args.weight_decay)
    else: 
        raise ValueError("Unknown optimizer.")

    # learning rate decay and epsilon scheduling
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    eps_schedule = np.linspace(args.starting_epsilon, args.epsilon, args.schedule_length)

    clas_err_min = 1
    robust_err_min = 1
    flag = False    # indicate whether we can find a proper clasifier

    for t in range(args.epochs):
        lr_scheduler.step(epoch=max(t-len(eps_schedule), 0))
        if t < len(eps_schedule) and args.starting_epsilon is not None: 
            epsilon = float(eps_schedule[t])
        else:
            epsilon = args.epsilon

        # standard training
        if args.method == 'baseline': 
            train_baseline(train_loader, model, opt, t, train_log, train_res, args.verbose)
            clas_err = evaluate_baseline(valid_loader, model, t, valid_res, args.verbose)

            if clas_err < clas_err_min:
                flag = True
                t_best = t
                clas_err_min = clas_err
                torch.save(model.state_dict(), model_path)

        # robust training for overall robustness
        elif args.method == 'overall_robust':
            train_robust(train_loader, model, opt, epsilon, t, train_log, train_res, 
                        args.verbose, l1_type=args.l1_train, bounded_input=True, **kwargs)
            clas_err, robust_err = evaluate_robust(valid_loader, model, args.epsilon, t, valid_res, 
                                    args.verbose, l1_type=args.l1_test, bounded_input=True, **kwargs)
            
            if clas_err <= args.thres and robust_err < robust_err_min and t >= args.schedule_length:
                flag = True
                t_best = t
                clas_err_best = clas_err
                robust_err_min = robust_err    
                torch.save(model.state_dict(), model_path)
        else:
            raise ValueError("Unknown type of training method.")

    print('==================== tuning results ====================')
    if flag == False:
        print('None of the epochs evaluated satisfy the criteria')
    else:
        if args.method == 'baseline':
            print('at epoch', t_best, 'achieves')
            print('lowest classification error:', '{:.2%}'.format(clas_err_min))
            print('baseline model:', t_best, '{:.2%}'.format(clas_err_min), file=best_res)
        elif args.method == 'overall_robust':
            print('at epoch', t_best, 'achieves')
            print('classification error:', '{:.2%}'.format(clas_err_best))
            print('lowest overall robust error:', '{:.2%}'.format(robust_err_min))
            print('overall robust model:', t_best, '{:.2%}'.format(clas_err_best), 
                    '{:.2%}'.format(robust_err_min), file=best_res)

        # evaluating the saved best model on the testing dataset
        model = select_model(args.model) 
        model.load_state_dict(torch.load(model_path))

        res_filepath = ('../results/'+args.proctitle)
        res_folder= os.path.dirname(res_filepath)
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)

        evaluate_test_clas_spec(test_loader, model, args.epsilon, res_filepath, 
                        args.verbose, l1_type=args.l1_test, bounded_input=True, **kwargs)
