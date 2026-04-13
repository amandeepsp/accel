#!/usr/bin/env python3
"""
Export quantized MNIST model for ACCEL CFU hardware deployment.

This script produces:
1. Quantized weights as int8 arrays
2. Per-layer quantization scales
3. Layer-to-layer requantization parameters with fixed-point conversions
4. Complete configuration for hardware implementation

Output: models/accel_hardware_package.json (or .npz for binary weights)
"""

import logging
import sys
import json
from pathlib import Path

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_fixed_point_multiplier(
    multiplier_fp32: float, max_shift: int = 31
) -> tuple:
    """
    Convert floating-point multiplier to fixed-point representation for hardware.

    For gemmlowp-style fixed-point:
    result = (multiplier_fixed_point * value) >> shift_amount

    Args:
        multiplier_fp32: Float multiplier (e.g., input_scale / output_scale)
        max_shift: Maximum bit shift (default 31 for int32)

    Returns:
        (fixed_point_mult_int32, shift_amount): Tuple ready for hardware
    """
    if multiplier_fp32 == 0:
        return (0, 0)

    # We want: multiplier_fp32 ≈ (fixed_point_mult / 2^shift)
    # Rearrange: fixed_point_mult ≈ multiplier_fp32 * 2^shift

    # Find shift that keeps fixed_point_mult in valid int32 range [-(2^31), (2^31)-1]
    # Start from shift=0 and increase until we fit
    shift = 0
    fixed_mult_fp = multiplier_fp32

    while fixed_mult_fp < (2**30) and shift < max_shift:
        fixed_mult_fp *= 2
        shift += 1

    # Convert to int32
    fixed_mult_int = int(np.round(fixed_mult_fp))

    # Clamp to valid int32 range
    fixed_mult_int = np.clip(fixed_mult_int, -(2**31), (2**31) - 1)

    logger.debug(
        f"Fixed-point multiplier: {multiplier_fp32:.8f} -> int={fixed_mult_int}, shift={shift}"
    )
    return (fixed_mult_int, shift)


def extract_scales_from_quantized_weight(weight):
    """Extract per-channel scales from a quantized weight tensor."""
    try:
        # LinearActivationQuantizedTensor stores original AQT
        if hasattr(weight, "original_weight_tensor"):
            original_weight = weight.original_weight_tensor
            # AffineQuantizedTensor has tensor_impl with scale
            if hasattr(original_weight, "tensor_impl"):
                tensor_impl = original_weight.tensor_impl
                if hasattr(tensor_impl, "__dict__") and "scale" in tensor_impl.__dict__:
                    scale_tensor = tensor_impl.scale
                    if isinstance(scale_tensor, torch.Tensor):
                        if scale_tensor.numel() > 1:
                            return scale_tensor.detach().cpu().tolist(), "per_channel"
                        else:
                            return float(scale_tensor.detach().cpu()), "scalar"
    except Exception as e:
        logger.debug(f"Error extracting scales: {e}")

    return None, None


def extract_int8_weights(weight):
    """Extract int8 quantized data from a quantized weight tensor."""
    try:
        if hasattr(weight, "original_weight_tensor"):
            original_weight = weight.original_weight_tensor
            if hasattr(original_weight, "tensor_impl"):
                tensor_impl = original_weight.tensor_impl
                if (
                    hasattr(tensor_impl, "__dict__")
                    and "int_data" in tensor_impl.__dict__
                ):
                    int_data = tensor_impl.int_data
                    return int_data.detach().cpu().numpy().astype(np.int8)
    except Exception as e:
        logger.debug(f"Error extracting int8 weights: {e}")

    return None


def main():
    """Main export routine."""
    device = "cpu"
    logger.info(f"Loading model on device: {device}")

    # Load checkpoint
    model_path = "models/mnist_int8_torchao.pt"
    logger.info(f"Loading checkpoint from {model_path}")

    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    config = checkpoint["config"]

    logger.info(f"Model config: {config}")
    logger.info(f"Model is quantized: {checkpoint.get('quantized', False)}")

    # Extract quantization info and weights
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTING WEIGHTS AND QUANTIZATION PARAMETERS")
    logger.info("=" * 60)

    model_state = checkpoint["model_state"]
    layer_data = {}
    linear_layers = []

    # Process all layers
    for key in sorted(model_state.keys()):
        if "weight" in key and not key.endswith("bias"):
            layer_name = key.replace(".weight", "")
            tensor = model_state[key]

            logger.info(f"\n{layer_name}:")
            logger.info(f"  Shape: {tensor.shape}")

            # Extract int8 weights
            int8_weights = extract_int8_weights(tensor)
            if int8_weights is not None:
                logger.info(f"  Int8 weights extracted: {int8_weights.shape}")
            else:
                logger.info(f"  Could not extract int8 weights, using FP32")
                int8_weights = tensor.cpu().numpy().astype(np.float32)

            # Extract scales
            scales, scale_type = extract_scales_from_quantized_weight(tensor)
            if scales is not None:
                if scale_type == "per_channel":
                    mean_scale = float(np.mean(scales))
                    logger.info(f"  Per-channel scales: {len(scales)} channels")
                    logger.info(
                        f"    Mean: {mean_scale:.6f}, Range: [{min(scales):.6f}, {max(scales):.6f}]"
                    )
                    layer_data[layer_name] = {
                        "weights": int8_weights.tolist(),  # Convert to list for JSON
                        "weight_shape": list(int8_weights.shape),
                        "scales_per_channel": scales,
                        "scale_mean": mean_scale,
                    }
                else:
                    logger.info(f"  Scale (scalar): {scales:.6f}")
                    layer_data[layer_name] = {
                        "weights": int8_weights.tolist(),
                        "weight_shape": list(int8_weights.shape),
                        "scale": float(scales),
                    }
            else:
                logger.info(f"  No quantization info")
                layer_data[layer_name] = {
                    "weights": int8_weights.tolist(),
                    "weight_shape": list(int8_weights.shape),
                    "scale": 1.0,
                }

            linear_layers.append(layer_name)

    # Compute layer-to-layer requantization parameters
    logger.info("\n" + "=" * 60)
    logger.info("COMPUTING REQUANTIZATION PARAMETERS")
    logger.info("=" * 60)

    requant_export = {}
    for i in range(len(linear_layers) - 1):
        curr_layer = linear_layers[i]
        next_layer = linear_layers[i + 1]

        curr_info = layer_data[curr_layer]
        next_info = layer_data[next_layer]

        # Get mean scales
        curr_scale = curr_info.get("scale_mean", curr_info.get("scale", 1.0))
        next_scale = next_info.get("scale_mean", next_info.get("scale", 1.0))

        # Compute multiplier
        multiplier_fp32 = curr_scale / next_scale
        fixed_mult_int, shift = compute_fixed_point_multiplier(multiplier_fp32)

        key = f"layer_{i}_to_{i + 1}"
        requant_export[key] = {
            "from_layer": curr_layer,
            "to_layer": next_layer,
            "input_scale": float(curr_scale),
            "output_scale": float(next_scale),
            "multiplier_fp32": float(multiplier_fp32),
            "multiplier_fixed_point": int(fixed_mult_int),
            "shift_amount": int(shift),
            "input_zero_point": 0,
            "output_zero_point": 0,
        }

        logger.info(f"\n{key}:")
        logger.info(
            f"  {curr_layer} ({len(layer_data[curr_layer].get('scales_per_channel', [curr_scale]))} ch) "
            f"→ {next_layer} ({len(layer_data[next_layer].get('scales_per_channel', [next_scale]))} ch)"
        )
        logger.info(f"  Input scale: {curr_scale:.8f}")
        logger.info(f"  Output scale: {next_scale:.8f}")
        logger.info(f"  Multiplier (FP32): {multiplier_fp32:.8f}")
        logger.info(f"  Fixed-point: int={fixed_mult_int}, shift={shift}")

    # Create final hardware package
    logger.info("\n" + "=" * 60)
    logger.info("CREATING HARDWARE DEPLOYMENT PACKAGE")
    logger.info("=" * 60)

    hardware_package = {
        "metadata": {
            "format_version": "1.0",
            "quantization_scheme": "int8_per_channel_symmetric",
            "hardware_target": "ACCEL_CFU",
        },
        "model_config": {
            "input_size": 784,
            "hidden_sizes": config["hidden_sizes"],
            "num_classes": config["num_classes"],
            "quantized": checkpoint.get("quantized", False),
        },
        "layers": layer_data,
        "requantization_parameters": requant_export,
    }

    # Save as JSON (note: weights are large, could use .npz for binary)
    output_path = "models/accel_hardware_package.json"
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving hardware package to {output_file}")
    with open(output_file, "w") as f:
        json.dump(hardware_package, f, indent=2)

    logger.info(f"✓ Hardware package saved")
    logger.info(f"  File size: {output_file.stat().st_size / (1024 * 1024):.2f} MB")
    logger.info(f"  Layers: {len(hardware_package['layers'])}")
    logger.info(
        f"  Requantization param sets: {len(hardware_package['requantization_parameters'])}"
    )

    # Also save a compact version with just parameters (no weights)
    compact_package = {
        "metadata": hardware_package["metadata"],
        "model_config": hardware_package["model_config"],
        "layers": {
            name: {
                "weight_shape": info["weight_shape"],
                "scale_mean": info.get("scale_mean", info.get("scale", 1.0)),
                "is_per_channel": "scales_per_channel" in info,
            }
            for name, info in layer_data.items()
        },
        "requantization_parameters": requant_export,
    }

    compact_path = "models/accel_hardware_config.json"
    with open(compact_path, "w") as f:
        json.dump(compact_package, f, indent=2)

    logger.info(f"\nCompact config saved to {compact_path}")
    logger.info(f"  (Contains parameters only, no weights)")

    logger.info("\n✓ Hardware export complete!")
    logger.info(f"  Full package: {output_path}")
    logger.info(f"  Config only: {compact_path}")


if __name__ == "__main__":
    main()
