#!/usr/bin/env -S uv run python
"""Run Layer 0 (1x784x256) on Verilator sim in isolation and verify against CPU."""

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

log = logging.getLogger("layer0_isolated")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Layer 0 isolation test on Verilator sim")
    parser.add_argument("--tcp", default="tcp://127.0.0.1:21450")
    parser.add_argument("--onnx", default=str(REPO_ROOT / "models/out/mnist_int8.onnx"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--driver-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        log.error("ONNX model not found: %s", onnx_path)
        return 1

    import onnx
    model = onnx.load(str(onnx_path))
    weights = {}
    for init in model.graph.initializer:
        weights[init.name] = onnx.numpy_helper.to_array(init)

    w0 = weights["net.1.weight_quantized"]  # [256, 784] int8
    b0 = weights["net.1.bias_quantized"]    # [256] int32
    log.info("Weights: L0=%s b0=%s", w0.shape, b0.shape)

    rng = np.random.default_rng(args.seed)
    input_img = rng.integers(-128, 128, size=(1, 784), dtype=np.int8)

    # --- CPU reference for Layer 0 ---
    n0 = 256
    mult0 = np.ones(n0, dtype=np.int32) * (1 << 30)
    shift0 = np.zeros(n0, dtype=np.int32)
    cpu_acc0 = input_img.astype(np.int32) @ w0.T.astype(np.int32)
    cpu_h = cpu_requantize(cpu_acc0, b0, mult0, shift0, 0, 0, 127)
    log.info("CPU Layer 0: shape=%s min=%d max=%d", cpu_h.shape, cpu_h.min(), cpu_h.max())

    # --- Sim Layer 0 (isolated) ---
    transport = TcpTransport(args.tcp, timeout_s=args.driver_timeout)
    try:
        sim_h = run_gemm_on_sim(
            transport, input_img, w0.T.astype(np.int8),
            bias=b0.astype(np.int32), multiplier=mult0, shift=shift0,
            output_offset=0, activation_min=0, activation_max=127,
        )
    finally:
        transport.close()

    log.info("Sim Layer 0:  shape=%s min=%d max=%d", sim_h.shape, sim_h.min(), sim_h.max())

    # --- Compare ---
    delta = np.abs(cpu_h.astype(np.int32) - sim_h.astype(np.int32))
    m, n = cpu_h.shape

    within_0 = int((delta == 0).sum())
    within_1 = int((delta <= 1).sum())
    exact_pct = 100 * within_0 / (m * n)
    within1_pct = 100 * within_1 / (m * n)
    max_delta = int(delta.max())
    log.info("Layer 0: max|Δ|=%d, exact=%.1f%%, within±1=%.1f%%", max_delta, exact_pct, within1_pct)

    # Detailed per-channel diagnostics for the last N-tile (channels 248-255)
    tile = 8
    last_n_tile_start = (n // tile - 1) * tile
    log.info("")
    log.info("--- Per-channel detail: last N-tile (ch %d..%d) ---", last_n_tile_start, n - 1)
    for c in range(max(0, last_n_tile_start - tile), n):
        cpu_val = int(cpu_h[0, c])
        sim_val = int(sim_h[0, c])
        d = abs(cpu_val - sim_val)
        marker = " <-- MISMATCH" if d > 0 else ""
        log.info("  ch%3d: cpu=%4d  sim=%4d  |Δ|=%d%s", c, cpu_val, sim_val, d, marker)

    # List all mismatches
    big_mismatches = []
    for c in range(n):
        d = abs(int(cpu_h[0, c]) - int(sim_h[0, c]))
        if d > 1:
            big_mismatches.append((c, int(cpu_h[0, c]), int(sim_h[0, c]), d))

    if big_mismatches:
        log.info("")
        log.info("--- Large mismatches (|Δ| > 1): %d channels ---", len(big_mismatches))
        for c, cpu_v, sim_v, d in big_mismatches:
            log.info("  ch%3d: cpu=%4d  sim=%4d  |Δ|=%d", c, cpu_v, sim_v, d)

    log.info("")

    # Check: are our suspected failure channels still present?
    suspect_channels = [251, 255]
    for c in suspect_channels:
        cpu_v = int(cpu_h[0, c])
        sim_v = int(sim_h[0, c])
        d = abs(cpu_v - sim_v)
        if d > 0:
            log.info("SUSPECT ch%d: cpu=%d sim=%d |Δ|=%d (BUG CONFIRMED)", c, cpu_v, sim_v, d)
        else:
            log.info("SUSPECT ch%d: cpu=%d sim=%d MATCH (interesting...)", c, cpu_v, sim_v)

    if max_delta > 1:
        log.info("FAIL: max|Δ|=%d > tolerance", max_delta)
        return 1
    else:
        log.info("PASS: all channels within ±1")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
