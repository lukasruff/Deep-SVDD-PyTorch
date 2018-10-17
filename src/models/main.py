from .mnist_LeNet import MNIST_LeNet
from .cifar10_LeNet import CIFAR10_LeNet


def build_model(model_name):
    """Builds the model."""

    implemented_models = ('mnist_LeNet', 'cifar10_LeNet')
    assert model_name in implemented_models

    model = None

    if model_name == 'mnist_LeNet':
        model = MNIST_LeNet()

    if model_name == 'cifar10_LeNet':
        model = CIFAR10_LeNet()

    return model
