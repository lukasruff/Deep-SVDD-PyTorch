from .torchvision_dataset import TorchvisionDataset
import torchvision.datasets as tvds
import torchvision.transforms as transforms


class CIFAR10_Dataset(TorchvisionDataset):

    def __init__(self, root: str):
        super().__init__(root)
        self.train_set = tvds.CIFAR10(root=self.root, train=True, download=True)
        self.test_set = tvds.CIFAR10(root=self.root, train=False, download=True)
