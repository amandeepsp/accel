"""Loom compiler: TVM Relax integration for the CFU accelerator.

This package contains the out-of-tree TVM lowering pipeline:
- patterns: DPL pattern matching for quantized matmul composites
- codegen: Lowering loom regions to runtime calls
- relax: Pass entrypoints (tiling, partitioning, lowering)
- runtime: Host runtime bridge (packed funcs, memory upload, execution)
- quant_utils: Epilogue multiplier/shift derivation
"""

from .codegen import lower_loom_regions, get_composite_constants
from .patterns import partition_for_loom_cfu, make_matmul_requant_pattern
from .relax import lower_pipeline
from .runtime import (
    LoomRuntime,
    RuntimeConfig,
    SerialTransport,
    TcpTransport,
    register_runtime_functions,
)

__all__ = [
    "lower_loom_regions",
    "get_composite_constants",
    "partition_for_loom_cfu",
    "make_matmul_requant_pattern",
    "lower_pipeline",
    "LoomRuntime",
    "RuntimeConfig",
    "SerialTransport",
    "TcpTransport",
    "register_runtime_functions",
]
