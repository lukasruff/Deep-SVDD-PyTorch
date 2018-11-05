import torch
import numpy as np


def get_target_label_idx(labels, targets):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape[1:]))

    mean = torch.sum(x, dim=tuple(range(1, x.dim()))) / n_features  # mean over all features (pixels) per sample
    x -= mean.view((x.shape[0],) + (1,) * (x.dim() - 1))

    if scale == 'l1':
        x_scale = torch.sum(torch.abs(x), dim=tuple(range(1, x.dim()))) / n_features

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2, dim=tuple(range(1, x.dim())))) / n_features

    x /= x_scale.view((x.shape[0],) + (1,) * (x.dim() - 1))

    return x
