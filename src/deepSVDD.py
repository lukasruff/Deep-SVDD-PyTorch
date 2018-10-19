from base.base_dataset import BaseADDataset
from models.main import build_model
from optim.deepSVDD_trainer import DeepSVDDTrainer


class DeepSVDD(object):
    """A class for the Deep SVDD method.

    Attributes:
        dataset_name: A string indicating the name of the dataset to load.
        data: The Dataset.
        model_name: A string indicating the name of the model to use.
        model: The model.
    """

    def __init__(self, objective: str = 'one-class', nu: float = 0.1):
        """Inits DeepSVDD with data and model."""

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.nu = nu
        self.R = 0  # hypersphere radius R
        self.c = None  # hypersphere center c

        self.model_name = None
        self.model = None  # neural network model \phi

        self.trainer = None
        self.optimizer_name = None

    def set_model(self, model_name):
        """Builds the model."""
        self.model_name = model_name
        self.model = build_model(model_name)

    def train(self, dataset: BaseADDataset, optimizer_name, lr: float = 0.001, n_epochs: int = 100,
              batch_size: int = 128):
        """Trains the model on the training data."""

        self.optimizer_name = optimizer_name

        self.trainer = DeepSVDDTrainer(self.objective, optimizer_name,
                                       lr=lr, n_epochs=n_epochs, batch_size=batch_size, nu=self.nu)
        self.model = self.trainer.train(dataset, self.model)

    def test(self, dataset: BaseADDataset):
        """Tests the model on the test data."""

        self.trainer.test(dataset, self.model)
