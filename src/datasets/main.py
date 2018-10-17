from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset


def load_dataset(dataset_name):
    """Loads the dataset."""

    data_root = '../data'

    implemented_datasets = ('mnist', 'cifar10')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_root)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_root)

    return dataset
