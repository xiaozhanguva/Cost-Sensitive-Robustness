# Cost-Sensitive Robustness against Adversarial Examples
The goal of this project [link to the original ArXiv paper](https://arxiv.org/pdf/1810.09225.pdf):
* For any pre-designed cost matrix, define the cost-sensitive robustness for neural network classifiers
* Develop a general method for training certified cost-sensitive robust classifier against 
![l_infty](https://latex.codecogs.com/gif.latex?%5Cell_%5Cinfty) adversarial attack
* Compare with existing certified overall robust classifier on MNIST and CIFAR-10 datasets

# Installation & Usage
The code was developed using Python3 on [Anaconda](https://www.anaconda.com/download/#linux)
* Install Pytorch 0.4.1: 
```text
conda update -n base conda && conda install pytorch=0.4.1 torchvision -c pytorch -y
```
* Install convex_adversarial package developed by Eric Wong and Zico Kolter
[[see details]](https://github.com/locuslab/convex_adversarial/tree/master/convex_adversarial):
```text
pip install --upgrade pip && pip install convex_adversarial==0.3.5 -I --user torch==0.4.1
```
* Install other dependencies:
```text
pip install torch waitGPU setproctitle
```

* Examples for training the cost-sensitive robust classifier:
  ```text
  cd examples && python mnist_task_spec.py --type real --category small-large --tuning coarse
  ```
  ```text
  cd examples && python cifar_task_spec.py --model large --type binary --category single_pair
  ```


# What is in this respository?
* ```examples```, including:
  * ```problems.py```: defines the dataloaders and neural network architectures for MNIST and CIFAR-10
  * ```trainer.py```: implements the detailed training and evaluation functions for different classifiers
  * ```mnist.py, cifar.py```: main functions for training overall robust classifier 
  * ```mnist_task_spec.py, cifar_task_spec.py```: main functions for training cost-sensitive robust classifier
  * ```heatmap.py```: implements functions for generating heatmap for any given cost matrix

* ```main_plot_overall.py```: produces the robust heatmap for overall robust classifier
* ```main_stats_mnist.py```: computes the summary statistics for both robust classifiers on MNIST
