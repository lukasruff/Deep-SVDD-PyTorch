class DeepSVDD(object):
    """A class for the Deep SVDD method.

    Attributes:
        dataset: A string indicating the dataset to load.
        model: A string indicating the model to use.
    """

    def __init__(self, dataset, model):
        """Inits DeepSVDD with dataset and model."""
        self.dataset = dataset
        self.model = model

    def load_data(self):
        """ Loads the dataset"""
        pass

    def train(self):
        """Trains the model"""
        pass
