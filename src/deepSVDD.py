from torch.utils.data import DataLoader
from datasets.main import load_dataset


class DeepSVDD(object):
    """A class for the Deep SVDD method.

    Attributes:
        dataset_name: A string indicating the name of the dataset to load.
        dataset: The Dataset.
        data: The DataLoader for the dataset.
        model: A string indicating the model to use.
    """

    def __init__(self, dataset_name, model):
        """Inits DeepSVDD with data and model."""
        self.dataset_name = dataset_name
        self.dataset = None
        self.data = None
        self.model = model

    def load_data(self):
        """ Loads the data"""
        self.dataset = load_dataset(self.dataset_name)
        self.data = DataLoader(self.dataset)
        pass

    def train(self):
        """Trains the model"""
        pass
