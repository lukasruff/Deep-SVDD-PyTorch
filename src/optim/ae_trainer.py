from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class AETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str, lr: float = 0.001, n_epochs: int = 150, batch_size: int = 128,
                 weight_decay: float = 1e-6, n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, batch_size, weight_decay, n_jobs_dataloader)

    def train(self, dataset: BaseADDataset, ae_net: BaseNet):

        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        # TODO: Implement choice of different losses (MSE, CrossEntropyLoss)
        criterion = nn.MSELoss()

        # Set optimizer
        # TODO: Implement choice of different optimizers ('sgd', 'momentum', 'nesterov', etc.)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)  # Adam optimizer for now

        # Training
        print('Starting training.')
        ae_net.train()
        for epoch in range(self.n_epochs):

            loss_epoch = 0.0
            n_batches = 0
            for data in train_loader:
                inputs, _, _ = data

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = ae_net(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            # print epoch statistics
            print('[Epoch %d] loss: %.8f' % (epoch + 1, loss_epoch / n_batches))

        print('Finished Training.')

        return ae_net

    def test(self, dataset: BaseADDataset, ae_net: BaseNet):

        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Testing
        print('Starting testing.')
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                outputs = ae_net(inputs)
                # compute reconstruction errors
                scores = torch.sum(criterion(outputs, inputs), dim=tuple(range(outputs.dim()))[1:])

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
