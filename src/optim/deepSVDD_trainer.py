from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import torch
import torch.optim as optim
import numpy as np


class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, optimizer_name: str, lr: float = 0.001, n_epochs: int = 150, batch_size: int = 128,
                 nu: float = 0.1, n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, batch_size, n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.nu = nu
        self.c = None
        self.R = None

    def train(self, dataset: BaseADDataset, net: BaseNet):

        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer
        # TODO: Implement choice of different optimizers ('sgd', 'momentum', 'nesterov', etc.)
        optimizer = optim.Adam(net.parameters(), lr=self.lr)  # Adam optimizer for now

        # Initialize hypersphere center c
        self.c = init_center_c(train_loader, net)

        # Initialize hypersphere radius R with 0 for soft-boundary DeepSVDD
        if self.objective == 'soft-boundary':
            self.R = torch.tensor(0.0)

        # Training
        print('Starting training.')
        net.train()
        for epoch in range(self.n_epochs):

            loss_epoch = 0.0
            n_batches = 0
            for data in train_loader:
                inputs, _, _ = data

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if self.objective == 'soft-boundary':
                    self.R.data = torch.tensor(get_radius(dist, self.nu))

                loss_epoch += loss.item()
                n_batches += 1

            # print epoch statistics
            print('[Epoch %d] loss: %.8f' % (epoch + 1, loss_epoch / n_batches))

        print('Finished Training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):

        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        print('Starting testing.')
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.data.numpy().tolist(),
                                            labels.data.numpy().tolist(),
                                            scores.data.numpy().tolist()))

        indices, labels, scores = zip(*idx_label_score)
        indices = list(indices)  # convert from tuple to list
        labels = np.array(labels)
        scores = np.array(scores)

        auc = roc_auc_score(labels[indices], scores)
        print('Test set AUC: {:.2f}%'.format(100. * auc))

        print('Finished testing.')


def init_center_c(train_loader: DataLoader, net: BaseNet, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""

    print('Initializing center c...')

    n_samples = 0
    c = torch.zeros(net.rep_dim)

    net.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            inputs, _, _ = data
            outputs = net(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # TODO: Make sure elements of center c are not initialized too close to 0 using eps?

    print("Center c initialized.")

    return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(dist.data.numpy(), 1-nu)
