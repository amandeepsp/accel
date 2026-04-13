"""MNIST dataset utilities."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloaders(batch_size=32, data_dir="./data"):
    """
    Load MNIST dataset and return train/test dataloaders.

    Args:
        batch_size: Batch size for dataloaders
        data_dir: Directory to download/load MNIST data

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Normalize to MNIST standard
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader
