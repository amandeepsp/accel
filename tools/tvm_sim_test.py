#!/usr/bin/env -S uv run python
"""MNIST inference on Verilator simulation via the TVM runtime transport.

Loads the int8 ONNX model weights, runs both GEMM layers on the sim,
and compares against a CPU reference that models the hardware epilogue.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tvm"))

from runtime import TcpTransport
from shared.reference import cpu_requantize
from shared.sim_harness import run_gemm_on_sim

log = logging.getLogger("tvm_sim_test")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MNIST inference on Verilator sim")
    parser.add_argument("--tcp", default="tcp://127.0.0.1:21450")
    parser.add_argument("--onnx", default=str(REPO_ROOT / "models/out/mnist_int8.onnx"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-tolerance", type=int, default=1)
    parser.add_argument("--driver-timeout", type=float, default=1800.0,
                        help="TCP transport timeout in seconds (default 1800)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        log.error("ONNX model not found: %s", onnx_path)
        log.error("Run: uv run python -m models.mnist")
        return 1

    # Load weights
    import onnx
    model = onnx.load(str(onnx_path))
    weights = {}
    for init in model.graph.initializer:
        weights[init.name] = onnx.numpy_helper.to_array(init)

    w0 = weights["net.1.weight_quantized"]  # [256, 784] int8
    b0 = weights["net.1.bias_quantized"]    # [256] int32
    w1 = weights["net.3.weight_quantized"]  # [10, 256] int8
    b1 = weights["net.3.bias_quantized"]    # [10] int32

    log.info("Weights: L0=%s b0=%s  L1=%s b1=%s", w0.shape, b0.shape, w1.shape, b1.shape)

    # Random test input
    rng = np.random.default_rng(args.seed)
    input_img = rng.integers(-128, 128, size=(1, 784), dtype=np.int8)

    # --- CPU reference ---
    n0, n1 = 256, 10
    mult0 = np.ones(n0, dtype=np.int32) * (1 << 30)
    shift0 = np.zeros(n0, dtype=np.int32)
    cpu_acc0 = input_img.astype(np.int32) @ w0.T.astype(np.int32)
    cpu_h = cpu_requantize(cpu_acc0, b0, mult0, shift0, 0, 0, 127)

    mult1 = np.ones(n1, dtype=np.int32) * (1 << 30)
    shift1 = np.zeros(n1, dtype=np.int32)
    cpu_acc1 = cpu_h.astype(np.int32) @ w1.T.astype(np.int32)
    cpu_out = cpu_requantize(cpu_acc1, b1, mult1, shift1, 0, -128, 127)
    log.info("CPU: pred=%d logits=%s", int(cpu_out.argmax()),
             str(cpu_out.flatten().tolist()))

    # --- Sim inference ---
    transport = TcpTransport(args.tcp, timeout_s=args.driver_timeout)
    try:
        sim_h = run_gemm_on_sim(
            transport, input_img, w0.T.astype(np.int8),
            bias=b0.astype(np.int32), multiplier=mult0, shift=shift0,
            output_offset=0, activation_min=0, activation_max=127,
        )

        sim_out = run_gemm_on_sim(
            transport, sim_h, w1.T.astype(np.int8),
            bias=b1.astype(np.int32), multiplier=mult1, shift=shift1,
            output_offset=0, activation_min=-128, activation_max=127,
        )
    finally:
        transport.close()

    log.info("Sim: pred=%d logits=%s", int(sim_out.argmax()),
             str(sim_out.flatten().tolist()))

    # Compare
    tol = args.verify_tolerance
    delta_h = np.abs(cpu_h.astype(np.int32) - sim_h.astype(np.int32))
    delta_out = np.abs(cpu_out.astype(np.int32) - sim_out.astype(np.int32))

    log.info("Layer 0: max|Δ|=%d, within±%d: %.1f%%",
             int(delta_h.max()), tol, 100 * (delta_h <= tol).mean())
    log.info("Layer 1: max|Δ|=%d, within±%d: %.1f%%",
             int(delta_out.max()), tol, 100 * (delta_out <= tol).mean())

    same_pred = int(cpu_out.argmax()) == int(sim_out.argmax())
    pass_tol = int(delta_h.max()) <= tol and int(delta_out.max()) <= tol

    if same_pred or pass_tol:
        log.info("PASS")
        return 0
    else:
        log.error("FAIL: mismatched prediction or tolerance exceeded")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
