"""Shared MNIST ONNX → Relax → loom-VM pipeline runner.

Used by both the Verilator simulation test (``tests/test_sim.py``) and the
hardware test (``tests/test_hw_pipeline.py``) so the two paths exercise *exactly* the
same TVM lowering + dispatch flow against their respective transports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compiler import LoomRuntime, lower_pipeline, register_runtime_functions

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = REPO_ROOT / "models/out/mnist_int8.onnx"


@dataclass
class MnistResult:
    cpu_out: np.ndarray   # (1, 10) fp32 — onnxruntime reference
    loom_out: np.ndarray  # (1, 10) fp32 — TVM Relax VM dispatching to loom
    registered: list[str]
    max_delta: float
    cpu_pred: int
    loom_pred: int


def run_mnist_via_vm(
    runtime: LoomRuntime,
    *,
    onnx_path: Path = DEFAULT_ONNX,
    seed: int = 42,
) -> MnistResult:
    """Run MNIST end-to-end: onnxruntime CPU vs TVM Relax VM with loom-backed externs.

    The lowered module's external symbols are registered against ``runtime`` so
    the VM dispatches GEMM tiles through whatever transport the runtime owns
    (TCP for sim, serial for hardware).
    """
    import onnx
    import onnxruntime as ort
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    onnx_path = Path(onnx_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"missing ONNX model: {onnx_path}")

    rng = np.random.default_rng(seed)
    input_img = rng.integers(-128, 128, size=(1, 784), dtype=np.int8)
    vm_input = np.zeros((1, 1, 28, 28), dtype=np.float32)
    vm_input[0, 0, :, :] = input_img.astype(np.float32).reshape(28, 28)

    sess = ort.InferenceSession(str(onnx_path))
    cpu_out = sess.run(None, {sess.get_inputs()[0].name: vm_input})[0]

    model_proto = onnx.load(str(onnx_path))
    mod = from_onnx(
        model_proto,
        shape_dict={"input": [1, 1, 28, 28]},
        keep_params_in_input=False,
    )
    lowered = lower_pipeline(mod)
    registered = register_runtime_functions(lowered, runtime=runtime)

    ex = relax.build(lowered, target="c")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    loom_out = vm["main"](vm_input).numpy()

    delta = np.abs(cpu_out.flatten() - loom_out.flatten())
    return MnistResult(
        cpu_out=cpu_out,
        loom_out=loom_out,
        registered=registered,
        max_delta=float(delta.max()),
        cpu_pred=int(cpu_out.argmax()),
        loom_pred=int(loom_out.argmax()),
    )
