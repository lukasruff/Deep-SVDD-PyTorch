from base.base_dataset import BaseADDataset
from networks.main import build_network, build_autoencoder
from optim.deepSVDD_trainer import DeepSVDDTrainer
from optim.ae_trainer import AETrainer


class DeepSVDD(object):
    """A class for the Deep SVDD method.

    Attributes:
        objective: A string specifying the Deep SVDD objective (either 'one-class' or 'soft-boundary').
        nu: Deep SVDD hyperparameter nu (must be 0 < nu <= 1).
        R: Hypersphere radius R.
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network \phi.
        ae_net: The autoencoder network corresponding to \phi for network weights pretraining.
        trainer: DeepSVDDTrainer to train a Deep SVDD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SVDD network.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
    """

    def __init__(self, objective: str = 'one-class', nu: float = 0.1):
        """Inits DeepSVDD with one of the two objectives and hyperparameter nu."""

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.nu = nu
        self.R = 0  # hypersphere radius R
        self.c = None  # hypersphere center c

        self.net_name = None
        self.net = None  # neural network \phi

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

    def set_network(self, net_name):
        """Builds the neural network \phi."""
        self.net_name = net_name
        self.net = build_network(net_name)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              batch_size: int = 128, weight_decay: float = 1e-6, n_jobs_dataloader: int = 0):
        """Trains the Deep SVDD model on the training data."""

        self.optimizer_name = optimizer_name
        self.trainer = DeepSVDDTrainer(self.objective, optimizer_name, lr=lr, n_epochs=n_epochs, batch_size=batch_size,
                                       weight_decay=weight_decay, n_jobs_dataloader=n_jobs_dataloader, nu=self.nu)
        self.net = self.trainer.train(dataset, self.net)

    def pretrain(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 batch_size: int = 128, weight_decay: float = 1e-6, n_jobs_dataloader: int = 0):

        self.ae_net = build_autoencoder(self.net_name)
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, batch_size=batch_size,
                                    weight_decay=weight_decay, n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)
        self.ae_trainer.test(dataset, self.ae_net)

    def test(self, dataset: BaseADDataset):
        """Tests the Deep SVDD model on the test data."""

        self.trainer.test(dataset, self.net)

    def save_model(self, xp_path):
        """Save Deep SVDD model to xp_path."""
        # Make save_ae an argument
        pass

    def load_model(self, model_path):
        """Load Deep SVDD model from model_path."""
        # Write such that this could be an AE network or Deep SVDD network
        # Make load_ae an argument
        pass
