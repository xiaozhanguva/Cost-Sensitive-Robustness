import waitGPU
waitGPU.wait(utilization=20, available_memory=10000, interval=10)

from examples.trainer import *
import examples.problems as pblm
import setproctitle

if __name__ == '__main__':
	args = pblm.argparser(prefix='mnist', method='task_spec_robust', opt='adam', 
							starting_epsilon=0.05, epsilon=0.2)
	kwargs = pblm.args2kwargs(args)
	setproctitle.setproctitle('python')

	# train-validation split
	_, _, test_loader = pblm.mnist_loaders(batch_size=args.batch_size, path='./data',
	 									   ratio=args.ratio, seed=args.seed)

	model = pblm.mnist_model().cuda()
	num_classes = model[-1].out_features

	# specify the task and the corresponding class semantic
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

	print('==================== baseline model ====================')
	model_overall_path = ('models/'+args.prefix+'/overall_robust/'+os.path.basename(args.proctitle)+'.pth')
	model.load_state_dict(torch.load(model_overall_path))
	clas_err_overall, robust_cost_overall = evaluate_test(test_loader, model, args.epsilon, 
													input_mat, args.type, args.verbose,
													l1_type=args.l1_test, bounded_input=True, **kwargs)

	print('==================== our model ====================')
	model_task_spec_path = ('models/'+folder_path+'/fine/'+os.path.basename(args.proctitle)+'.pth')
	model.load_state_dict(torch.load(model_task_spec_path))
	clas_err_task_spec, robust_cost_task_spec = evaluate_test(test_loader, model, args.epsilon, 
													input_mat, args.type, args.verbose,
													l1_type=args.l1_test, bounded_input=True, **kwargs)

	# save the result
	if not os.path.exists('results/'+folder_path):
		os.makedirs('results/'+folder_path)
	res_filepath = os.path.join('results/'+folder_path, os.path.basename(args.proctitle))
	res_log = open(res_filepath+'_res.txt', "w")

	if args.type == 'binary':
		print('baseline model:', '{:.2%}'.format(clas_err_overall), 
						'{:.2%}'.format(robust_cost_overall), file=res_log)
		print('our model:', '{:.2%}'.format(clas_err_task_spec), 
						'{:.2%}'.format(robust_cost_task_spec), file=res_log)
	else:	# real-valued
		print('baseline model:', '{:.2%}'.format(clas_err_overall), 
						'{:.3f}'.format(robust_cost_overall), file=res_log)
		print('our model:', '{:.2%}'.format(clas_err_task_spec), 
						'{:.3f}'.format(robust_cost_task_spec), file=res_log)



