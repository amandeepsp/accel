#!/usr/bin/env python3
"""
Training and quantization script for int8 MNIST using torchao with requantization params.

This trains a model and then quantizes it to actual int8 using torchao,
collecting requantization parameters suitable for hardware deployment (gemmlowp-style).

Usage:
    python -m models.train_int8_torchao --epochs 10 --batch-size 32
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.torchao_quantization import (
    Int8QuantizedMNISTRequant,
    compare_fp32_int8_requant,
)
from models.mnist_utils import get_mnist_dataloaders

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(train_loader), 100 * correct / total


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Train and quantize int8 MNIST model with torchao (requantization params)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[512, 256],
        help="Hidden layer sizes (default: 512 256)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/mnist_int8_torchao.pt",
        help="Path to save quantized model",
    )
    parser.add_argument(
        "--export-hw",
        type=str,
        default=None,
        help="Export for hardware to this path (JSON)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for MNIST data",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip quantization, just save FP32 model",
    )

    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Starting Int8 MNIST Training with Torchao (Requantization Params)")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Hidden sizes: {args.hidden_sizes}")
    logger.info(f"Device: {device}")

    # Create model
    model = Int8QuantizedMNISTRequant(hidden_sizes=args.hidden_sizes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Load data
    train_loader, test_loader = get_mnist_dataloaders(args.batch_size, args.data_dir)

    # Training
    logger.info("Starting training")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_acc = evaluate(model, test_loader, device)

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%"
        )

    logger.info("Training complete")

    # Save FP32 model before quantization
    fp32_model = Int8QuantizedMNISTRequant(hidden_sizes=args.hidden_sizes)
    fp32_model.load_state_dict(model.state_dict())

    if not args.no_quantize:
        logger.info(
            "Starting int8 quantization with torchao (dynamic activation + int8 weight)"
        )
        try:
            model.quantize_int8(device=device)
            logger.info("Quantization successful")

            # Compare FP32 vs Int8
            logger.info("Comparing FP32 vs Int8 models")
            comparison = compare_fp32_int8_requant(
                fp32_model, model, test_loader, device
            )

            logger.info(f"Quantization metrics:")
            logger.info(f"  FP32 Accuracy: {comparison['fp32_accuracy']:.2f}%")
            logger.info(f"  Int8 Accuracy: {comparison['int8_accuracy']:.2f}%")
            logger.info(f"  Agreement: {comparison['agreement']:.2f}%")

        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            import traceback

            traceback.print_exc()
            logger.info("Saving FP32 model only")

    # Save model
    import os

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    model.save_quantized(args.save_path)
    logger.info(f"Model saved to {args.save_path}")

    # Export for hardware if requested
    if args.export_hw:
        import json

        logger.info(f"Exporting for hardware to {args.export_hw}")
        export_data = model.export_for_hardware()

        # Convert numpy arrays to lists for JSON serialization
        export_json = {
            "config": export_data["config"],
            "quantized_weights": {},
            "requantization_params": export_data["requantization_params"],
        }

        for name, weight_data in export_data["quantized_weights"].items():
            export_json["quantized_weights"][name] = {
                "weight": weight_data["weight"].tolist(),
                "bias": weight_data["bias"].tolist()
                if weight_data["bias"] is not None
                else None,
                "weight_scale": weight_data["weight_scale"],
                "weight_zero_point": weight_data["weight_zero_point"],
            }

        os.makedirs(os.path.dirname(args.export_hw) or ".", exist_ok=True)
        with open(args.export_hw, "w") as f:
            json.dump(export_json, f, indent=2)

        logger.info(f"Hardware export saved to {args.export_hw}")


if __name__ == "__main__":
    main()
