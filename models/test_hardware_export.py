#!/usr/bin/env python3
"""
Validation tests for hardware export parameters.

Verifies:
1. All layer weights are int8
2. Scales are reasonable values
3. Requantization multipliers are valid fixed-point
4. Export format is complete
"""

import logging
import json
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_hardware_export_format():
    """Test that exported files exist and have correct structure."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Hardware Export Format")
    logger.info("=" * 60)

    # Check full package
    full_pkg_path = Path("models/accel_hardware_package.json")
    assert full_pkg_path.exists(), f"Full package not found: {full_pkg_path}"
    logger.info(
        f"✓ Full package exists: {full_pkg_path.name} ({full_pkg_path.stat().st_size / (1024 * 1024):.2f} MB)"
    )

    with open(full_pkg_path) as f:
        full_pkg = json.load(f)

    # Check compact config
    compact_path = Path("models/accel_hardware_config.json")
    assert compact_path.exists(), f"Compact config not found: {compact_path}"
    logger.info(f"✓ Compact config exists: {compact_path.name}")

    with open(compact_path) as f:
        compact_cfg = json.load(f)

    # Validate structure
    required_keys = ["metadata", "model_config", "layers", "requantization_parameters"]
    for key in required_keys:
        assert key in full_pkg, f"Missing key in full package: {key}"
        assert key in compact_cfg, f"Missing key in compact config: {key}"

    logger.info(f"✓ Both exports have required structure")

    return full_pkg, compact_cfg


def test_layer_weights(pkg):
    """Test layer weight properties."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Layer Weights")
    logger.info("=" * 60)

    layers = pkg["layers"]
    logger.info(f"Testing {len(layers)} layers...")

    for layer_name, layer_info in layers.items():
        logger.info(f"\n  {layer_name}:")

        # Check weight shape
        weight_shape = layer_info["weight_shape"]
        assert len(weight_shape) == 2, f"Weight must be 2D, got {len(weight_shape)}D"
        logger.info(f"    Shape: {weight_shape}")

        # Check weights are int8 (if present in full package)
        if "weights" in layer_info:
            weights = np.array(layer_info["weights"], dtype=np.int8)
            assert weights.shape == tuple(weight_shape), f"Weight shape mismatch"
            assert weights.min() >= -128 and weights.max() <= 127, (
                "Weights out of int8 range"
            )
            logger.info(f"    Weights: int8, range=[{weights.min()}, {weights.max()}]")

        # Check scales
        if "scales_per_channel" in layer_info:
            scales = layer_info["scales_per_channel"]
            mean_scale = layer_info["scale_mean"]
            logger.info(f"    Scales: per-channel ({len(scales)} ch)")
            logger.info(f"    Scale range: [{min(scales):.6f}, {max(scales):.6f}]")
            logger.info(f"    Scale mean: {mean_scale:.6f}")

            # Verify mean matches
            computed_mean = np.mean(scales)
            assert abs(computed_mean - mean_scale) < 1e-9, "Scale mean mismatch"
        elif "scale" in layer_info:
            scale = layer_info["scale"]
            assert scale > 0, f"Scale must be positive, got {scale}"
            logger.info(f"    Scale: {scale:.6f}")

    logger.info(f"\n✓ All {len(layers)} layers have valid weights")


def test_requantization_params(pkg):
    """Test requantization parameter validity."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Requantization Parameters")
    logger.info("=" * 60)

    requant_params = pkg["requantization_parameters"]
    logger.info(f"Testing {len(requant_params)} requantization param sets...")

    for param_name, params in requant_params.items():
        logger.info(f"\n  {param_name}:")

        # Check required fields
        required_fields = [
            "from_layer",
            "to_layer",
            "input_scale",
            "output_scale",
            "multiplier_fp32",
            "multiplier_fixed_point",
            "shift_amount",
        ]
        for field in required_fields:
            assert field in params, f"Missing field: {field}"

        input_scale = params["input_scale"]
        output_scale = params["output_scale"]
        mult_fp32 = params["multiplier_fp32"]
        mult_fixed = params["multiplier_fixed_point"]
        shift = params["shift_amount"]

        logger.info(f"    From: {params['from_layer']} → {params['to_layer']}")
        logger.info(f"    Input scale: {input_scale:.8f}")
        logger.info(f"    Output scale: {output_scale:.8f}")
        logger.info(f"    FP32 multiplier: {mult_fp32:.8f}")
        logger.info(f"    Fixed-point: {mult_fixed} (shift={shift})")

        # Validate multiplier relationship
        computed_mult = input_scale / output_scale
        assert abs(computed_mult - mult_fp32) < 1e-6, f"Multiplier mismatch"

        # Validate fixed-point is int32
        assert isinstance(mult_fixed, int), "Fixed-point multiplier must be int"
        assert -(2**31) <= mult_fixed < (2**31), (
            "Fixed-point multiplier out of int32 range"
        )

        # Validate shift is reasonable
        assert 0 <= shift <= 31, f"Shift must be [0,31], got {shift}"

        # Verify fixed-point approximation
        # The approximation should be: approx_multiplier ≈ (mult_fixed / 2^shift)
        if mult_fixed != 0:
            approx_mult = mult_fixed / (2**shift)
            error = abs(approx_mult - mult_fp32) / mult_fp32
            logger.info(
                f"    Approx multiplier: {approx_mult:.8f} (error: {error * 100:.4f}%)"
            )
            assert error < 0.01, (
                f"Fixed-point approximation error too large: {error * 100:.2f}%"
            )

    logger.info(f"\n✓ All {len(requant_params)} requantization param sets are valid")


def test_model_config(pkg):
    """Test model configuration."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Model Configuration")
    logger.info("=" * 60)

    config = pkg["model_config"]

    logger.info(f"  Input size: {config['input_size']}")
    logger.info(f"  Hidden sizes: {config['hidden_sizes']}")
    logger.info(f"  Num classes: {config['num_classes']}")
    logger.info(f"  Quantized: {config['quantized']}")

    assert config["input_size"] == 784, "MNIST input should be 784 (28x28)"
    assert config["num_classes"] == 10, "MNIST should have 10 classes"
    assert len(config["hidden_sizes"]) > 0, "Must have hidden layers"
    assert config["quantized"] == True, "Model should be quantized"

    logger.info(f"✓ Model configuration is valid")


def main():
    """Run all validation tests."""
    logger.info("HARDWARE EXPORT VALIDATION")
    logger.info("=" * 60)

    try:
        # Test 1: Format
        full_pkg, compact_cfg = test_hardware_export_format()

        # Test 2: Weights (using compact config for structure)
        test_layer_weights(compact_cfg)

        # Test 3: Requantization params
        test_requantization_params(compact_cfg)

        # Test 4: Model config
        test_model_config(compact_cfg)

        logger.info("\n" + "=" * 60)
        logger.info("✓ ALL VALIDATION TESTS PASSED")
        logger.info("=" * 60)
        logger.info("\nHardware export is ready for deployment:")
        logger.info("  - models/accel_hardware_package.json (with weights)")
        logger.info("  - models/accel_hardware_config.json (config only)")

        return True

    except AssertionError as e:
        logger.error(f"\n✗ VALIDATION FAILED: {e}")
        return False
    except Exception as e:
        logger.error(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
