from mnist import select_model

import problems as pblm
from trainer import *
import setproctitle

if __name__ == "__main__": 
    args = pblm.argparser(prefix='mnist', method='task_spec_robust', opt='adam', 
                          starting_epsilon=0.05, epsilon=0.2, thres=0.035)
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
              'Train estimation: {1}\t\t'
              'Test estimation: {2}'.format(
                args.l1_proj, args.l1_train, args.l1_test), end='\n')

    # train-validation split
    train_loader, valid_loader, test_loader = pblm.mnist_loaders(args.batch_size, args.ratio, args.seed)
    model = select_model(args.model)
    num_classes = model[-1].out_features

    for X,y in train_loader: 
        kwargs = pblm.args2kwargs(model, args, X=Variable(X.cuda()))
        break

    # specify the task and construct the corresponding cost matrix 
    folder_path = os.path.dirname(args.proctitle)
    if args.type == 'binary':
        input_mat = np.zeros((num_classes,num_classes), dtype=np.int)
        if args.category == 'single_seed':
            seed_clas = 9
            input_mat[seed_clas, :] = np.ones(num_classes)
            input_mat[seed_clas, seed_clas] = 0
            folder_path += '/class_'+str(seed_clas)
        else:
            raise ValueError("Unknown category of binary task.")

    elif args.type == 'real':
        input_mat = np.zeros((num_classes,num_classes), dtype=np.float)
        if args.category == 'small-large':
            for i in range(num_classes):
                for j in range(num_classes):
                    if i > j:
                        continue
                    else:
                        dist = np.absolute(i-j)
                        input_mat[i,j] = dist*dist
        else:
            raise ValueError("Unknown category of real-valued task.")

    else:
        raise ValueError("Unknown type of cost matrix.")

    print('==================== cost matrix ====================')
    print(input_mat)

    # define the searching grid for alpha
    print('====================', args.tuning, 'tuning ====================')
    if args.tuning == 'coarse':
        alpha_arr = np.float_power(10, np.arange(-1,2))
        # raise NotImplementedError()
    elif args.tuning == 'fine':
        alpha_select = 0.01
        alpha_arr = np.float_power(2, np.arange(-1,2))*alpha_select
        # raise NotImplementedError()
    else:
        raise ValueError("Unknown type of tuning method.")
    print('Searching grid for alpha:', alpha_arr)

    # define the saved_log and model file path
    folder_path += '/'+args.tuning
    saved_folder = '../saved_log/'+folder_path
    model_folder = '../models/'+folder_path
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    saved_filepath = os.path.join(saved_folder, os.path.basename(args.proctitle))
    model_path = os.path.join(model_folder, os.path.basename(args.proctitle)+'.pth')

    if args.tuning == 'coarse':
        res_log = open(saved_filepath + '_res_log.txt', "w")

    # train the task-specific robust model and tuning the alpha      
    robust_cost_best = np.inf
    flag = False    # indicator for finding the desirable classifier
    for k in range(len(alpha_arr)):
        alpha = alpha_arr[k].item()
        print('Current stage: alpha = '+str(alpha))
        clas_err_min = 1.0

        train_log = open(saved_filepath + '_train_log_alpha_' + str(alpha) + '.txt', "w")
        train_res = open(saved_filepath + '_train_res_alpha_' + str(alpha) + '.txt', "w")
        valid_res = open(saved_filepath + '_valid_res_alpha_' + str(alpha) + '.txt', "w")

        # specify the model and the optimizer
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        model = select_model(args.model) 

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

        for t in range(args.epochs):
            lr_scheduler.step(epoch=max(t-len(eps_schedule), 0))
            if t < len(eps_schedule) and args.starting_epsilon is not None: 
                epsilon = float(eps_schedule[t])
            else:
                epsilon = args.epsilon

            train_robust_task_spec(train_loader, model, opt, epsilon, t, train_log, 
                                train_res, args.verbose, input_mat, args.type, alpha, 
                                l1_type=args.l1_train, bounded_input=True, **kwargs)
            clas_err, robust_cost = evaluate_robust_task_spec(valid_loader, model, args.epsilon, t, 
                                                valid_res, args.verbose, input_mat, args.type, alpha, 
                                                l1_type=args.l1_test, bounded_input=True, **kwargs)
            
            if args.tuning == 'coarse':
                if clas_err < clas_err_min and t >= args.schedule_length:    
                    t_min = t
                    clas_err_min = clas_err
                    robust_cost_min = robust_cost
                    torch.save(model.state_dict(), model_path) 

            elif args.tuning == 'fine':
                if clas_err <= args.thres and robust_cost < robust_cost_best and t >= args.schedule_length:
                    flag = True
                    alpha_best = alpha
                    t_best = t
                    clas_err_best = clas_err
                    robust_cost_best = robust_cost
                    torch.save(model.state_dict(), model_path) 

        if args.tuning == 'coarse':
            # evaluate the model on testing data for each alpha
            model.load_state_dict(torch.load(model_path))
            clas_err_test, robust_cost_test = evaluate_test(test_loader, model, args.epsilon, 
                                                    input_mat, args.type, args.verbose,
                                                    l1_type=args.l1_test, bounded_input=True, **kwargs)

            print('==================== intermediate results for coarse tuning ====================')
            if args.type == 'binary':
                print('Alpha {0}\t'
                    'Iteration {1}\t'
                    'Error(valid) {err_valid:.2%}\t'
                    'Robust cost(valid) {rcost_valid:.2%}\t'
                    'Error(test) {err_test:.2%}\t'
                    'Robust cost(test) {rcost_test:.2%}\t'.format(
                    alpha, t_min, err_valid=clas_err_min, rcost_valid=robust_cost_min,
                    err_test=clas_err_test, rcost_test=robust_cost_test, end='\n'))
                print(alpha, t_min, '{:.2%}'.format(clas_err_min), '{:.2%}'.format(robust_cost_min),
                            '{:.2%}'.format(clas_err_test), '{:.2%}'.format(robust_cost_test), file=res_log)
            else:
                print('Alpha {0}\t'
                    'Iteration {1}\t'
                    'Error(valid) {err_valid:.2%}\t'
                    'Robust cost(valid) {rcost_valid:.3f}\t'
                    'Error(test) {err_test:.2%}\t'
                    'Robust cost(test) {rcost_test:.3f}\t'.format(
                    alpha, t_min, err_valid=clas_err_min, rcost_valid=robust_cost_min,
                    err_test=clas_err_test, rcost_test=robust_cost_test, end='\n'))
                print(alpha, t_min, '{:.2%}'.format(clas_err_min), '{:.3f}'.format(robust_cost_min),
                            '{:.2%}'.format(clas_err_test), '{:.3f}'.format(robust_cost_test), file=res_log)

            res_log.flush()

    if args.tuning == 'fine':
        print('==================== final tuning results for fine tuning ====================')
        if flag == False:
            print('None of the regularization parameters satisfy the criteria.')
        else:
            print('best alpha:', alpha_best)
            print('at epoch', t_best, 'achieves')
            print('classification error:', '{:.2%}'.format(clas_err_best))
            best_res = open(saved_filepath + "_best_res.txt", "w")

            if args.type == 'binary':
                print('cost-sensitive robust error:', '{:.2%}'.format(robust_cost_best))
                print(alpha_best, t_best, '{:.2%}'.format(clas_err_best), 
                        '{:.2%}'.format(robust_cost_best), file=best_res)
            else: 
                print('average cost:', '{:.3f}'.format(robust_cost_best))
                print(alpha_best, t_best, '{:.2%}'.format(clas_err_best), 
                        '{:.3f}'.format(robust_cost_best), file=best_res)

            # evaluating the saved best model on the testing dataset            
            res_folder = ('../results/' + folder_path)
            if not os.path.exists(res_folder):
                os.makedirs(res_folder)
            res_filepath = os.path.join(res_folder, os.path.basename(args.proctitle))
            
            model.load_state_dict(torch.load(model_path))    
            evaluate_test_clas_spec(test_loader, model, args.epsilon, res_filepath, args.verbose,
                                    l1_type=args.l1_test, bounded_input=True,**kwargs)



        
        

                