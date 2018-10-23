from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
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


def build_ae_model(model_name):
    """Builds the respective autoencoder model."""

    implemented_models = ('mnist_LeNet')
    assert model_name in implemented_models

    ae_model = None

    if model_name == 'mnist_LeNet':
        ae_model = MNIST_LeNet_Autoencoder()

    return ae_model
