from cifar import select_model
import problems as pblm
from trainer import *

import setproctitle
import random

if __name__ == "__main__": 
    args = pblm.argparser(prefix='cifar', method='task_spec_robust', epsilon=0.03486, l1_proj=50, 
                        l1_train='median', starting_epsilon=0.001, opt='sgd', lr=0.05, thres=0.35)
    setproctitle.setproctitle('python')
    print("threshold for classification error: {:.1%}".format(args.thres))

    print('Matrix type: {0}\t\t'
          'Category: {1}\t\t'
          'Epoch number: {2}\t\t'
          'Targeted epsilon: {3}\t\t'
          'Starting epsilon: {4}\t\t'
          'Sechduled length: {5}'.format(
            args.type, args.category, args.epochs, 
            args.epsilon, args.starting_epsilon, args.schedule_length), end='\n')
    if args.l1_proj is not None:
        print('Projection vectors: {0}\t\t'
              'Train estimate: {1}\t\t'
              'Test estimate: {2}'.format(
                args.l1_proj, args.l1_train, args.l1_test), end='\n')

    # train-validation split
    train_loader, _, _ = pblm.cifar_loaders(batch_size=args.batch_size, ratio=args.ratio, seed=args.seed)
    _, valid_loader, test_loader = pblm.cifar_loaders(batch_size=1, ratio=args.ratio, seed=args.seed)
    model = select_model(args.model)
    num_classes = model[-1].out_features

    for X,y in train_loader: 
        kwargs = pblm.args2kwargs(model, args, X=Variable(X.cuda()))
        break

    # specify the task and the corresponding class semantic
    folder_path = os.path.dirname(args.proctitle)
    if args.type == 'binary':
        input_mat = np.zeros((num_classes,num_classes), dtype=np.int)
        if args.category == 'single_pair':
            seed_clas = 6
            targ_clas = 2
            input_mat[seed_clas,targ_clas] = 1
            folder_path += '/pair_'+str(seed_clas)+'_'+str(targ_clas)
        else:
            raise ValueError("Unknown category of binary task.")

    elif args.type == 'real':
        input_mat = np.zeros((num_classes,num_classes), dtype=np.float)
        if args.category == 'vehicle':
            ani_ind = [2,3,4,5,6,7]
            veh_ind = [0,1,8,9]
            for i in veh_ind:
                for j in ani_ind:
                    input_mat[i,j] = 10
                for j in veh_ind:
                    if i != j:
                        input_mat[i,j] = 1
        else:
            raise ValueError("Unknown category of real-valued task.")

    else:
        raise ValueError("Unknown type of cost matrix.")

    print('==================== cost matrix ====================')
    print(input_mat)

    # define the saved_log and model file path
    saved_folder = '../saved_log/'+folder_path
    model_folder = '../models/'+folder_path
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    saved_filepath = os.path.join(saved_folder, os.path.basename(args.proctitle))
    model_path = os.path.join(model_folder, os.path.basename(args.proctitle)+'.pth')

    # define the searching grid for alpha
    alpha_arr = [0.1, 1.0, 10]
    print('Searching grid for alpha:', alpha_arr)
    # raise NotImplementedError()

    # train the task-specific robust model and tuning the alpha      
    robust_cost_best = np.inf 
    flag = False    # indicator for finding the desirable classifier
    res_log = open(saved_filepath + '_res_log.txt', "w")
    
    for k in range(len(alpha_arr)):
        alpha = alpha_arr[k]
        print('Current stage: alpha = '+str(alpha))

        train_log = open(saved_filepath + '_train_log_alpha_' + str(alpha) + '.txt', "w")
        train_res = open(saved_filepath + '_train_res_alpha_' + str(alpha) + '.txt', "w")

        # specify the model and the optimizer
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(0)
        np.random.seed(0)
        model = select_model(args.model) 

        if args.opt == 'adam': 
            opt = optim.Adam(model.parameters(), lr=args.lr)
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

            train_robust_task_spec(train_loader, model, opt, epsilon, t, train_log, train_res, 
                                args.verbose, input_mat, args.type, alpha, l1_type=args.l1_train, 
                                bounded_input=False, clip_grad=1, **kwargs)

        # evaluate the last model on validation dataset for each alpha
        print('==================== validating ====================')
        clas_err_valid, robust_cost_valid = evaluate_test(valid_loader, model, args.epsilon, 
                                                    input_mat, args.type, verbose=len(valid_loader), 
                                                    l1_type=args.l1_test, bounded_input=False, **kwargs)

        print('==================== intermediate results for alpha tuning ====================')
        if args.type == 'binary':
            print('Alpha {0}\t\t'
                  'Error(valid) {err_valid:.2%}\t\t'
                  'Robust cost(valid) {rcost_valid:.2%}\t\t'.format(
                    alpha, err_valid=clas_err_valid, rcost_valid=robust_cost_valid, end='\n'))
            print(alpha, '{:.2%}'.format(clas_err_valid), 
                    '{:.2%}'.format(robust_cost_valid), file=res_log)
        else:   # real-valued
            print('Alpha {0}\t\t'
                  'Error(valid) {err_valid:.2%}\t\t'
                  'Robust cost(valid) {rcost_valid:.3f}\t\t'.format(
                    alpha, err_valid=clas_err_valid, rcost_valid=robust_cost_valid, end='\n'))
            print(alpha, '{:.2%}'.format(clas_err_valid), 
                    '{:.3f}'.format(robust_cost_valid), file=res_log)
        res_log.flush()

        # save the model from the last training epoch
        if clas_err_valid <= args.thres and robust_cost_valid < robust_cost_best:
            flag = True
            alpha_best = alpha
            clas_err_best = clas_err_valid
            robust_cost_best = robust_cost_valid
            torch.save(model.state_dict(), model_path)

    print('==================== final tuning results ====================')
    if flag == False:
        print('None of the regularization parameters satisfy the criteria')
    else:
        print('best alpha:', alpha_best)
        print('at the last epoch achieves')
        print('classification error:', '{:.2%}'.format(clas_err_best))
        best_res = open(saved_filepath + "_best_res.txt", "w")

        if args.type == 'binary':        
            print('cost-sensitive robust error:', '{:.2%}'.format(robust_cost_best))
            print(alpha_best, '{:.2%}'.format(clas_err_best), 
                    '{:.2%}'.format(robust_cost_best), file=best_res)
        else:   # real-valued
            print('average cost:', '{:.3f}'.format(robust_cost_best))
            print(alpha_best, '{:.2%}'.format(clas_err_best), 
                    '{:.3f}'.format(robust_cost_best), file=best_res)

        # evaluating the saved best model on the testing dataset
        model.load_state_dict(torch.load(model_path))
        res_folder = ('../results/' + folder_path)
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_filepath = os.path.join(res_folder, os.path.basename(args.proctitle))

        evaluate_test_clas_spec(test_loader, model, args.epsilon, res_filepath, verbose=len(test_loader),
                                l1_type=args.l1_test, bounded_input=False, **kwargs)

