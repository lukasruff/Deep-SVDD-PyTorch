from datasets.mnist import MNIST_Dataset
from datasets.cifar10 import CIFAR10_Dataset


def load_dataset(dataset_name):
    """Loads dataset."""

    implemented_datasets = ('mnist', 'cifar10')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset()

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset()

    return dataset
