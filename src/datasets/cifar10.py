from base.torchvision_dataset import TorchvisionDataset
from torch.utils.data import Subset
from .preprocessing import get_target_label_idx

import torchvision.datasets as tvds
import torchvision.transforms as transforms


class CIFAR10_Dataset(TorchvisionDataset):

    def __init__(self, root: str):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier

        normal_classes = tuple([0])
        outlier_classes = (1, 2, 3, 4, 5, 6, 7, 8, 9)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        target_transform = transforms.Lambda(lambda x: int(x in outlier_classes))

        self.train_set = tvds.CIFAR10(root=self.root, train=True, download=True,
                                      transform=transform, target_transform=target_transform)
        self.test_set = tvds.CIFAR10(root=self.root, train=False, download=True,
                                     transform=transform, target_transform=target_transform)

        # Subset train set to normal instances
        train_idx_normal = get_target_label_idx(self.train_set.train_labels, normal_classes)
        self.train_set = Subset(self.train_set, train_idx_normal)
