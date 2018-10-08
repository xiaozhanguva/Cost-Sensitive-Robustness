# Cost-Sensitive Robustness against Adversarial Examples
The goal of this project:
* For any pre-designed cost matrix, define the cost-sensitive robustness for neural network classifiers
* Develop a general method for training certified cost-sensitive robust classifier against 
![l_infty](https://latex.codecogs.com/gif.latex?%5Cell_%5Cinfty) adversarial attack
* Compare with the state-of-the-art [certified robust classifier](https://arxiv.org/abs/1805.12514) on MNIST and CIFAR-10 datasets


# What is in this respository?
* ```convex_adversarial```, including:
  * The convex_adversarial package developed by Eric Wong and Zico Kolter 
  [[see details]](https://github.com/locuslab/convex_adversarial)
  * ```robust_loss_task_spec()``` in dual_network.py: defines the metric and loss function for cost-sensitive robustness
  * ```calc_err_clas_spec()``` in dual_network.py: computes the pairwise classification and robust error

* ```examples```, including:
  * ```problems.py```: defines the dataloaders and neural network classifers for MNIST and CIFAR-10
  * ```trainer.py```: implements the training and evaluation procedures for robust classifiers
  * ```mnist.py, cifar.py```: main functions for training overall robust classifier 
  * ```mnist_task_spec.py```, cifar_task_spec.py: main functions for training cost-sensitive robust classifier
  * ```heatmap.py```: implements functions for generating heatmap for any given matrix

* ```main_plot_overall.py```: produces the robust heatmap for overall robust model
* ```main_stats_mnist.py```: computes the summary statistics for both robust classifiers on MNIST


# Installation & Usage
The code was developed on Python3 using [Anaconda](https://www.anaconda.com/download/#linux)
* Install Pytorch 0.4: 
```text
conda update -n base conda && conda install pytorch=0.4 torchvision -c pytorch
```
* Install dependencies: 
```text
pip install --upgrade pip && pip install torch waitGPU setproctitle
```
* Examples for training the cost-sensitive robust classifier:
  ```text
  python examples/mnist_task_spec.py --type real --category small-large --tuning coarse
  ```
  ```text
  python examples/cifar_task_spec.py --model large --type binary --category single_pair
  ```