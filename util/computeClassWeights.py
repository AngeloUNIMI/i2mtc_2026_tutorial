import os
import pickle
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def getClassCount(trainLoader, numClasses):

    classCounts = np.zeros(numClasses)

    for _, target in trainLoader:

        binC = np.bincount(target)

        if len(binC) < numClasses:  # no element for class 3
            binC = np.append(binC, np.zeros(numClasses-len(binC)))

        classCounts += binC


    return classCounts


def computeClassWeights(
    dataloader_train,
    cuda=False,
    weights_path=None,
    force_recompute=False,
    eps=1e-6,
    max_weight=100.0
):
    """
    Compute BCEWithLogitsLoss pos_weight for multi-label classification.

    Assumes targets have shape:
        (B, num_classes)

    Example target:
        [1, 0, 1, 0]

    Returns:
        weightsBCE with shape (num_classes,)
    """

    if weights_path is not None:
        if os.path.exists(weights_path) and not force_recompute:
            return torch.load(weights_path, map_location="cpu")

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    num_positive = None
    num_samples = 0

    for _, _, _, target in dataloader_train:
        target = target.to(device).float()

        if target.dim() != 2:
            raise ValueError(
                f"Expected target with shape (B, num_classes), got {tuple(target.shape)}"
            )

        if num_positive is None:
            num_classes = target.shape[1]
            num_positive = torch.zeros(
                num_classes,
                device=device,
                dtype=torch.float64
            )

        num_positive += target.sum(dim=0).double()
        num_samples += target.shape[0]

    if num_samples == 0:
        raise ValueError("No samples found in dataloader_train")

    num_negative = num_samples - num_positive

    weightsBCE = num_negative / (num_positive + eps)

    # Prevent extremely large weights if a class is very rare or absent
    weightsBCE = torch.clamp(weightsBCE, min=1.0, max=max_weight)

    weightsBCE = weightsBCE.float().cpu()

    if weights_path is not None:
        os.makedirs(os.path.dirname(weights_path), exist_ok=True) if os.path.dirname(weights_path) else None
        torch.save(weightsBCE, weights_path)

    return weightsBCE
