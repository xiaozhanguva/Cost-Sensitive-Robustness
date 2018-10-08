import waitGPU
waitGPU.wait(utilization=20, available_memory=10000, interval=60)

import problems as pblm
from trainer import *
import setproctitle
import random
    
def select_model(m): 
    if m == 'large': 
        # raise ValueError
        model = pblm.cifar_model_large().cuda()
    elif m == 'resnet': 
        model = pblm.cifar_model_resnet(N=args.resnet_N, factor=args.resnet_factor).cuda()
    else: 
        model = pblm.cifar_model().cuda() 
    return model

if __name__ == "__main__": 
    args = pblm.argparser(prefix='cifar', epsilon=0.03486, starting_epsilon=0.001, 
                        l1_proj=50, l1_train='median', opt='sgd', lr=0.05, ratio=0)
    setproctitle.setproctitle('python')
    print("saving file to {}".format(args.proctitle))

    saved_filepath = ('../saved_log/'+args.proctitle)
    model_filepath = os.path.dirname('../models/'+args.proctitle)
    if not os.path.exists(saved_filepath):
        os.makedirs(saved_filepath)
    if not os.path.exists(model_filepath):
        os.makedirs(model_filepath)
    model_path = ('../models/'+args.proctitle+'.pth')

    train_log = open(saved_filepath + '/train_log.txt', "w")
    train_res = open(saved_filepath + '/train_res.txt', "w")

    # generate dataloader for train with batch size=50
    train_loader, _ = pblm.cifar_loaders(batch_size=args.batch_size, ratio=args.ratio, seed=args.seed)
    # generate dataloader for test with batch size=1 (avoid GPU memory overflow)
    _, test_loader = pblm.cifar_loaders(batch_size=1, ratio=args.ratio, seed=args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(0)
    np.random.seed(0)
    model = select_model(args.model)

    for X,y in train_loader: 
        kwargs = pblm.args2kwargs(model, args, X)
        break

    if args.opt == 'adam': 
        opt = optim.Adam(model_path.parameters(), lr=args.lr)
    elif args.opt == 'sgd': 
        opt = optim.SGD(model.parameters(), lr=args.lr, 
                        momentum=args.momentum, weight_decay=args.weight_decay)
    else: 
        raise ValueError("Unknown optimizer")

    # learning rate decay and epsilon scheduling
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    eps_schedule = np.linspace(args.starting_epsilon, args.epsilon, args.schedule_length)

    for t in range(args.epochs):
        lr_scheduler.step(epoch=max(t-len(eps_schedule), 0))
        if t < len(eps_schedule) and args.starting_epsilon is not None: 
            epsilon = float(eps_schedule[t])
        else:
            epsilon = args.epsilon

        # standard training
        if args.method == 'baseline': 
            train_baseline(train_loader, model, opt, t, train_log, train_res, args.verbose)
    
        # robust training
        elif args.method == 'overall_robust':
            train_robust(train_loader, model, opt, epsilon, t, train_log, train_res, args.verbose, 
                        l1_type=args.l1_train, bounded_input=False, clip_grad=1, **kwargs)
        else:
            raise ValueError("Unknown type of training method.")

    # save the model from the last training epoch
    torch.save(model.state_dict(), model_path)
    model.load_state_dict(torch.load(model_path))

    res_filepath = ('../results/'+args.proctitle)
    res_folder= os.path.dirname(res_filepath)
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    # evaluate the test dataset (exact estimation)
    evaluate_test_clas_spec(test_loader, model, args.epsilon, res_filepath, verbose=len(test_loader), 
                            l1_type=args.l1_test, bounded_input=False, **kwargs)

