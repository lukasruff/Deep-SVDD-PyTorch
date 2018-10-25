from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset


def load_dataset(dataset_name, data_path):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path)

    return dataset
