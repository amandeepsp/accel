#!/usr/bin/env python
"""Run the MNIST TVM pipeline against a physical loom board.

Mirrors ``tests/test_sim.py::test_sim_pipeline`` but dispatches through a
``SerialTransport`` instead of TCP. Both share
``tests/mnist_pipeline.run_mnist_via_vm`` so the lowering + VM dispatch path
is identical.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from compiler import LoomRuntime, RuntimeConfig, SerialTransport

from tests.mnist_pipeline import DEFAULT_ONNX, run_mnist_via_vm

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ONNX -> Relax -> loom runtime MNIST inference on hardware.",
    )
    parser.add_argument("--port", default="/dev/ttyUSB1")
    parser.add_argument("--baud-rate", type=int, default=115200)
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    parser.add_argument("--lib", default=str(REPO_ROOT / "zig-out/lib/libaccel.so"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=0.1)
    args = parser.parse_args()

    transport = SerialTransport(
        port=args.port,
        baud_rate=args.baud_rate,
        lib_path=args.lib,
    )
    runtime = LoomRuntime(
        transport,
        RuntimeConfig(
            port=args.port,
            baud_rate=args.baud_rate,
            lib_path=args.lib,
        ),
    )

    try:
        result = run_mnist_via_vm(
            runtime,
            onnx_path=Path(args.onnx),
            seed=args.seed,
        )
    finally:
        runtime.close()

    print(f"lowered externs: {len(result.registered)}")
    for symbol in result.registered:
        print(f"  {symbol}")
    print(f"cpu output: {result.cpu_out.flatten().tolist()}")
    print(f"hw output:  {result.loom_out.flatten().tolist()}")
    print(f"max delta: {result.max_delta:.6f}")
    print(f"prediction: cpu={result.cpu_pred} hw={result.loom_pred}")

    if result.max_delta >= args.tolerance:
        raise AssertionError(
            f"max delta {result.max_delta:.6f} >= tolerance {args.tolerance}"
        )
    if result.loom_pred != result.cpu_pred:
        raise AssertionError(
            f"prediction mismatch: cpu={result.cpu_pred} hw={result.loom_pred}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
