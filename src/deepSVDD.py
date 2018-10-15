from datasets.main import load_dataset


class DeepSVDD(object):
    """A class for the Deep SVDD method.

    Attributes:
        dataset_name: A string indicating the name of the dataset to load.
        data: The Dataset.
        model_name: A string indicating the name of the model to use.
        model: The model.
    """

    def __init__(self, dataset_name, model_name):
        """Inits DeepSVDD with data and model."""
        self.dataset_name = dataset_name
        self.data = None
        self.model_name = model_name
        self.model = None

        # load data
        self.load_data()

    def load_data(self):
        """ Loads the data"""
        self.data = load_dataset(self.dataset_name)

    def train(self):
        """Trains the model"""
        pass
