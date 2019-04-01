# PyTorch Implementation of Deep SVDD
This repository provides a [PyTorch](https://pytorch.org/) implementation of the *Deep SVDD* method presented in our
ICML 2018 paper ”Deep One-Class Classification”.


## Citation and Contact
You find a PDF of the Deep One-Class Classification ICML 2018 paper at 
[http://proceedings.mlr.press/v80/ruff18a.html](http://proceedings.mlr.press/v80/ruff18a.html).

If you use our work, please also cite the paper:
```
@InProceedings{pmlr-v80-ruff18a,
  title     = {Deep One-Class Classification},
  author    = {Ruff, Lukas and Vandermeulen, Robert A. and G{\"o}rnitz, Nico and Deecke, Lucas and Siddiqui, Shoaib A. and Binder, Alexander and M{\"u}ller, Emmanuel and Kloft, Marius},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning},
  pages     = {4393--4402},
  year      = {2018},
  volume    = {80},
}
```

If you would like to get in touch, please contact [contact@lukasruff.com](mailto:contact@lukasruff.com).


## Abstract
> > Despite the great advances made by deep learning in many machine learning problems, there is a relative dearth of 
> > deep learning approaches for anomaly detection. Those approaches which do exist involve networks trained to perform 
> > a task other than anomaly detection, namely generative models or compression, which are in turn adapted for use in 
> > anomaly detection; they are not trained on an anomaly detection based objective. In this paper we introduce a new 
> > anomaly detection method—Deep Support Vector Data Description—, which is trained on an anomaly detection based
> > objective. The adaptation to the deep regime necessitates that our neural network and training procedure satisfy 
> > certain properties, which we demonstrate theoretically. We show the effectiveness of our method on MNIST and
> > CIFAR-10 image benchmark datasets as well as on the detection of adversarial examples of GTSRB stop signs.


## Installation
This code is written in `Python 3.7` and requires the packages listed in `requirements.txt`.

Clone the repository to your local machine and directory of choice:
```
git clone https://github.com/lukasruff/Deep-SVDD-PyTorch.git
```

To run the code, we recommend setting up a virtual environment, e.g. using `virtualenv` or `conda`:

### `virtualenv`
```
# pip install virtualenv
cd <path-to-Deep-SVDD-PyTorch-directory>
virtualenv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### `conda`
```
cd <path-to-Deep-SVDD-PyTorch-directory>
conda create --name myenv
source activate myenv
while read requirement; do conda install -n myenv --yes $requirement; done < requirements.txt
```


## Running experiments

We currently have implemented the MNIST ([http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)) and 
CIFAR-10 ([https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)) datasets and 
simple LeNet-type networks.

Have a look into `main.py` for all possible arguments and options.

### MNIST example
```
cd <path-to-Deep-SVDD-PyTorch-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# create folder for experimental output
mkdir log/mnist_test

# change to source directory
cd src

# run experiment
python main.py mnist mnist_LeNet ../log/mnist_test ../data --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 3;
```
This example trains a One-Class Deep SVDD model where digit 3 (`--normal_class 3`) is considered to be the normal class. Autoencoder
pretraining is used for parameter initialization.

### CIFAR-10 example
```
cd <path-to-Deep-SVDD-PyTorch-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# create folder for experimental output
mkdir log/cifar10_test

# change to source directory
cd src

# run experiment
python main.py cifar10 cifar10_LeNet ../log/cifar10_test ../data --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3;
```
This example trains a One-Class Deep SVDD model where cats (`--normal_class 3`) is considered to be the normal class. 
Autoencoder pretraining is used for parameter initialization.


## Examples

### MNIST
Example of the 32 most normal (left) and 32 most anomalous (right) test set examples per class on MNIST according to 
Deep SVDD anomaly scores.

![MNIST](imgs/mnist.png?raw=true "MNIST")

### CIFAR-10
Example of the 32 most normal (left) and 32 most anomalous (right) test set examples per class on CIFAR-10 according to 
Deep SVDD anomaly scores.

![CIFAR-10](imgs/cifar10.png?raw=true "CIFAR-10")


## License
MIT