from torch.utils.data import Dataset


class CIFAR10_Dataset(Dataset):
    """CIFAR-10 Dataset class."""

    def __init__(self):
        Dataset.__init__(self)

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
