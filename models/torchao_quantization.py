"""Int8 quantization using torchao with requantization parameters (gemmlowp-style)."""

import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import numpy as np

logger = logging.getLogger(__name__)


class QuantizationParams:
    """Represents quantization parameters for a tensor (gemmlowp-style)."""

    def __init__(self, scale: float, zero_point: int, dtype: torch.dtype = torch.int8):
        """
        Args:
            scale: Quantization scale factor
            zero_point: Zero point offset (-128 to 127 for int8)
            dtype: Data type (default int8)
        """
        self.scale = float(scale)
        self.zero_point = int(zero_point)
        self.dtype = dtype

    def __repr__(self):
        return f"QuantizationParams(scale={self.scale:.6f}, zero_point={self.zero_point}, dtype={self.dtype})"


class RequantizationParams:
    """
    Requantization parameters for layer-to-layer transformation.

    Follows gemmlowp convention:
    output_value = ((input_value * input_scale + accumulator) / output_scale) + output_zero_point
    """

    def __init__(
        self,
        input_scale: float,
        output_scale: float,
        input_zero_point: int = 0,
        output_zero_point: int = 0,
    ):
        """
        Args:
            input_scale: Scale of input tensor
            output_scale: Scale of output tensor
            input_zero_point: Zero point of input (default 0 for symmetric)
            output_zero_point: Zero point of output (default 0 for symmetric)
        """
        self.input_scale = float(input_scale)
        self.output_scale = float(output_scale)
        self.input_zero_point = int(input_zero_point)
        self.output_zero_point = int(output_zero_point)

        # Compute multiplier for fixed-point arithmetic
        # multiplier = input_scale / output_scale
        self.multiplier = self.input_scale / self.output_scale

    def __repr__(self):
        return (
            f"RequantizationParams("
            f"multiplier={self.multiplier:.6f}, "
            f"in_zp={self.input_zero_point}, "
            f"out_zp={self.output_zero_point})"
        )


def quantize_model_int8_dynamic(
    model: nn.Module,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Quantize model to int8 using torchao's dynamic activation + int8 weight quantization.

    Args:
        model: Model to quantize (FP32)
        device: Device to use

    Returns:
        Quantized model
    """
    from torchao.quantization import (
        Int8DynamicActivationInt8WeightConfig,
        quantize_,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    logger.info(
        "Applying int8 dynamic activation + int8 weight quantization via torchao"
    )

    # Use dynamic activation + int8 weight quantization
    quant_config = Int8DynamicActivationInt8WeightConfig()

    quantize_(model, quant_config)

    logger.info("Quantization complete via torchao")
    return model


def extract_quantization_params(model: nn.Module) -> Dict[str, QuantizationParams]:
    """
    Extract quantization parameters from a quantized model.

    Args:
        model: Quantized model

    Returns:
        Dictionary of quantization params per layer
    """
    params = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Try to extract quantization metadata
            if hasattr(module, "weight"):
                weight = module.weight

                # Try multiple ways to extract quantization info
                try:
                    if hasattr(weight, "q_scale"):
                        scale = weight.q_scale()
                        zero_point = (
                            weight.q_zero_point()
                            if hasattr(weight, "q_zero_point")
                            else 0
                        )
                        params[name] = QuantizationParams(float(scale), int(zero_point))
                except (NotImplementedError, RuntimeError, AttributeError):
                    pass

                # If weights are actual int8 tensors, infer scale from range
                if weight.dtype == torch.int8 and name not in params:
                    # Estimate scale: full int8 range [-128, 127] maps to actual range
                    weight_min = weight.min().float()
                    weight_max = weight.max().float()
                    weight_range = weight_max - weight_min
                    if weight_range > 0:
                        scale = weight_range / 255.0  # Full int8 range
                        zero_point = int(-weight_min / scale)
                        params[name] = QuantizationParams(float(scale), zero_point)

    return params


class Int8QuantizedMNISTRequant(nn.Module):
    """MNIST model with int8 quantization and requantization parameters (torchao-based)."""

    def __init__(self, hidden_sizes=None, num_classes=10):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 256]

        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes

        # Build FC layers
        layers = []
        prev_size = 784

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

        self.quantized = False

        # Store quantization parameters
        self.weight_quant_params = {}  # Weight quantization params
        self.requant_params = {}  # Layer requantization params (for layer-to-layer)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

    def quantize_int8(self, device=None, calibration_data=None):
        """
        Quantize model to int8 using torchao.

        Args:
            device: Device to use
            calibration_data: Optional calibration data for activation range estimation
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Quantize network
        self.network = quantize_model_int8_dynamic(self.network, device)

        # Extract weight quantization params
        self.weight_quant_params = extract_quantization_params(self.network)

        # If no params extracted (weights still FP32), estimate from weight statistics
        if not self.weight_quant_params:
            logger.info(
                "Estimating quantization parameters from FP32 weight statistics"
            )
            for name, module in self.network.named_modules():
                if isinstance(module, nn.Linear):
                    weight = module.weight.data
                    # Estimate int8 quantization params from weight range
                    weight_min = weight.min().item()
                    weight_max = weight.max().item()

                    # Symmetric quantization: range is [-127, 127]
                    abs_max = max(abs(weight_min), abs(weight_max))
                    if abs_max > 0:
                        scale = abs_max / 127.0
                        # For symmetric quantization, zero_point = 0
                        zero_point = 0
                    else:
                        scale = 1.0
                        zero_point = 0

                    self.weight_quant_params[name] = QuantizationParams(
                        scale, zero_point
                    )
                    logger.debug(f"  {name}: scale={scale:.6f}, zp={zero_point}")

        # Compute requantization parameters between layers
        linear_layers = [m for m in self.network.modules() if isinstance(m, nn.Linear)]
        linear_names = [
            name for name, m in self.network.named_modules() if isinstance(m, nn.Linear)
        ]

        for i in range(len(linear_layers) - 1):
            curr_layer_name = linear_names[i]
            next_layer_name = linear_names[i + 1]

            # Get scales
            curr_scale = self.weight_quant_params.get(
                curr_layer_name, QuantizationParams(1.0, 0)
            ).scale
            next_scale = self.weight_quant_params.get(
                next_layer_name, QuantizationParams(1.0, 0)
            ).scale

            # Requantization: output of curr layer (after ReLU) needs to match input scale of next layer
            # For now, assume activation scale = weight scale of current layer (can be refined with calibration)
            input_scale = curr_scale
            output_scale = next_scale

            requant = RequantizationParams(
                input_scale=input_scale,
                output_scale=output_scale,
                input_zero_point=0,
                output_zero_point=0,
            )
            self.requant_params[f"layer_{i}_to_{i + 1}"] = requant

        self.quantized = True
        logger.info(
            f"Collected {len(self.weight_quant_params)} weight quantization params"
        )
        logger.info(f"Computed {len(self.requant_params)} requantization param sets")

    def save_quantized(self, path: str):
        """Save quantized model with all parameters."""
        checkpoint = {
            "model_state": self.state_dict(),
            "config": {
                "hidden_sizes": self.hidden_sizes,
                "num_classes": self.num_classes,
            },
            "quantized": self.quantized,
            "weight_quant_params": {
                k: {"scale": v.scale, "zero_point": v.zero_point}
                for k, v in self.weight_quant_params.items()
            },
            "requant_params": {
                k: {
                    "multiplier": v.multiplier,
                    "input_scale": v.input_scale,
                    "output_scale": v.output_scale,
                    "input_zero_point": v.input_zero_point,
                    "output_zero_point": v.output_zero_point,
                }
                for k, v in self.requant_params.items()
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"Quantized model saved to {path}")

    @staticmethod
    def load_quantized(path: str):
        """Load quantized model."""
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        config = checkpoint["config"]
        model = Int8QuantizedMNISTRequant(**config)

        # Load quantized model state directly (torchao quantized tensors)
        try:
            model.load_state_dict(checkpoint["model_state"], strict=False)
        except RuntimeError:
            # If direct loading fails, model is already quantized
            # Just set model.network to the quantized network
            if "network" in checkpoint["model_state"]:
                model.network = checkpoint["model_state"]["network"]

        model.quantized = checkpoint.get("quantized", False)

        # Restore weight quant params
        for name, params_dict in checkpoint.get("weight_quant_params", {}).items():
            model.weight_quant_params[name] = QuantizationParams(
                params_dict["scale"], params_dict["zero_point"]
            )

        # Restore requant params
        for name, params_dict in checkpoint.get("requant_params", {}).items():
            model.requant_params[name] = RequantizationParams(
                params_dict["input_scale"],
                params_dict["output_scale"],
                params_dict["input_zero_point"],
                params_dict["output_zero_point"],
            )

        logger.info(f"Model loaded from {path} (quantized={model.quantized})")
        return model

    def export_for_hardware(self) -> Dict:
        """
        Export model in format suitable for hardware (like ACCEL CFU).

        Returns quantized weights + requantization parameters.
        """
        export_data = {
            "config": {
                "input_size": 784,
                "hidden_sizes": self.hidden_sizes,
                "num_classes": self.num_classes,
            },
            "quantized_weights": {},
            "requantization_params": {},
        }

        # Export weights as int8
        for name, module in self.network.named_modules():
            if isinstance(module, nn.Linear):
                # Get quantization params
                if name in self.weight_quant_params:
                    params = self.weight_quant_params[name]
                    scale = params.scale
                    zero_point = params.zero_point

                    # Quantize weights
                    weight_q = (
                        (module.weight.data / scale + zero_point)
                        .round()
                        .clamp(-128, 127)
                    )

                    export_data["quantized_weights"][name] = {
                        "weight": weight_q.cpu().numpy().astype(np.int8),
                        "bias": module.bias.data.cpu().numpy()
                        if module.bias is not None
                        else None,
                        "weight_scale": scale,
                        "weight_zero_point": zero_point,
                    }

        # Export requantization params
        for name, requant in self.requant_params.items():
            export_data["requantization_params"][name] = {
                "multiplier": requant.multiplier,
                "input_scale": requant.input_scale,
                "output_scale": requant.output_scale,
                "input_zero_point": requant.input_zero_point,
                "output_zero_point": requant.output_zero_point,
            }

        logger.info("Model exported for hardware with requantization parameters")
        return export_data


def compare_fp32_int8_requant(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Compare FP32 and int8 model accuracy.

    Args:
        fp32_model: FP32 model
        int8_model: Int8 quantized model
        test_loader: Test DataLoader
        device: Device to use

    Returns:
        Comparison metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fp32_model = fp32_model.to(device)
    int8_model = int8_model.to(device)

    fp32_model.eval()
    int8_model.eval()

    fp32_correct = 0
    int8_correct = 0
    agreement = 0
    total = 0

    logger.info("Comparing FP32 vs Int8 models")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            fp32_out = fp32_model(images)
            int8_out = int8_model(images)

            fp32_pred = fp32_out.argmax(dim=1)
            int8_pred = int8_out.argmax(dim=1)

            fp32_correct += (fp32_pred == labels).sum().item()
            int8_correct += (int8_pred == labels).sum().item()
            agreement += (fp32_pred == int8_pred).sum().item()

            total += labels.size(0)

    results = {
        "fp32_accuracy": 100 * fp32_correct / total,
        "int8_accuracy": 100 * int8_correct / total,
        "agreement": 100 * agreement / total,
    }

    logger.info(f"FP32 Accuracy: {results['fp32_accuracy']:.2f}%")
    logger.info(f"Int8 Accuracy: {results['int8_accuracy']:.2f}%")
    logger.info(f"Agreement: {results['agreement']:.2f}%")

    return results
