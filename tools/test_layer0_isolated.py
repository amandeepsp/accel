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

from runtime import TcpTransport, pack_weight_rows

log = logging.getLogger("layer0_isolated")

# ---------------------------------------------------------------------------
# Epilogue reference (matches hardware: SRDHM + RDBPOT + clamp)
# ---------------------------------------------------------------------------

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
MEM_ALIGN = 32


def align_up(v, a=MEM_ALIGN):
    return (v + a - 1) & -a


def ref_srdhm(a, b):
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    return max(INT32_MIN, min(INT32_MAX, (ab + nudge) >> 31))


def ref_rdbpot(x, exponent):
    if exponent == 0:
        return x
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + ((x >> 31) & 1)
    return (x >> exponent) + (1 if remainder > threshold else 0)


def cpu_requantize(acc, bias, multiplier, shift, output_offset, act_min, act_max):
    x = acc.astype(np.int64) + bias.astype(np.int64)
    out = np.zeros(acc.shape, dtype=np.int8)
    for r in range(acc.shape[0]):
        for c in range(acc.shape[1]):
            val = int(x[r, c])
            val = ref_srdhm(val, int(multiplier[c]))
            val = ref_rdbpot(val, int(shift[c]))
            val += output_offset
            out[r, c] = max(act_min, min(act_max, val))
    return out


# ---------------------------------------------------------------------------
# Full GEMM on sim (copied from tvm_sim_test.py)
# ---------------------------------------------------------------------------


def pack_input_tiles(matrix, tile):
    """Pack [M, K] into [K, tile] HW layout, zero-padding partial tiles."""
    _m, _k = matrix.shape
    chunks = []
    for m_base in range(0, _m, tile):
        ts = matrix[m_base: m_base + tile, :].T  # [K, tile_m]
        if ts.shape[1] < tile:
            pad = tile - ts.shape[1]
            ts = np.pad(ts, ((0, 0), (0, pad)), mode="constant")
        chunks.append(np.ascontiguousarray(ts))
    return np.concatenate(chunks).astype(np.int8).tobytes()


def run_gemm_on_sim(transport, lhs_orig, rhs, *,
                    bias=None, multiplier=None, shift=None,
                    output_offset=0, activation_min=-128, activation_max=127,
                    tile=8, cfu_word_bits=64, cfu_store_depth_words=512,
                    tensor_pool_base=0x40010100):
    """Execute a full GEMM (any M,N) on the sim via the transport."""

    from shared.ir import build_pipelined_gemm_program, plan_memory

    m_orig, k = lhs_orig.shape
    _k2, n = rhs.shape
    assert k == _k2

    if bias is None:
        bias = np.zeros(n, dtype=np.int32)
    if multiplier is None:
        multiplier = np.ones(n, dtype=np.int32) * (1 << 30)
    if shift is None:
        shift = np.zeros(n, dtype=np.int32)

    m = ((m_orig + tile - 1) // tile) * tile
    lhs = np.zeros((m, k), dtype=np.int8)
    lhs[:m_orig, :] = lhs_orig

    input_data = pack_input_tiles(lhs, tile)
    weight_data = pack_weight_rows(rhs)
    bias_data = bias.astype(np.int32).tobytes()
    mult_data = multiplier.astype(np.int32).tobytes()
    shift_data = shift.astype(np.int32).tobytes()
    output_size = m * n

    base = align_up(tensor_pool_base, MEM_ALIGN)
    input_addr = base
    weight_addr = input_addr + align_up(len(input_data))
    output_addr = weight_addr + align_up(len(weight_data))
    bias_addr = output_addr + align_up(output_size)
    mult_addr = bias_addr + align_up(n * 4)
    shift_addr = mult_addr + align_up(n * 4)

    layout = plan_memory(input_addr, weight_addr, output_addr,
                         bias_addr, mult_addr, shift_addr)
    program = build_pipelined_gemm_program(
        layout, m, k, n, tile,
        act_tensor_id=0, wgt_tensor_id=1, out_tensor_id=2,
        bias_id=3, mult_id=4, shift_id=5,
        cfu_word_bits=cfu_word_bits,
        cfu_store_depth_words=cfu_store_depth_words,
    )

    epi_bytes = bytearray(program)
    num_tensors = epi_bytes[5]
    i = 8 + num_tensors * 16
    patched = 0
    while i < len(epi_bytes):
        if epi_bytes[i] == 0x05:
            epi_bytes[i + 8] = output_offset & 0xFF
            epi_bytes[i + 9] = activation_min & 0xFF
            epi_bytes[i + 10] = activation_max & 0xFF
            patched += 1
            i += 12
        elif epi_bytes[i] in (0x01, 0x02):
            i += 8
        elif epi_bytes[i] in (0x03, 0x06):
            i += 4
        elif epi_bytes[i] == 0x04:
            i += 8
        else:
            log.warning("  unknown opcode 0x%02x at offset %d, stopping", epi_bytes[i], i)
            break
    log.info("  Patched %d set_epilogue instructions", patched)
    program = bytes(epi_bytes)

    log.info(
        "  GEMM %dx%dx%d (padded M=%d): uploading %d input + %d weight + %d epi bytes, "
        "program=%d bytes",
        m_orig, k, n, m,
        len(input_data), len(weight_data),
        len(bias_data) + len(mult_data) + len(shift_data),
        len(program),
    )
    t0 = time.monotonic()
    transport.write_mem(input_addr, input_data)
    transport.write_mem(weight_addr, weight_data)
    transport.write_mem(bias_addr, bias_data)
    transport.write_mem(mult_addr, mult_data)
    transport.write_mem(shift_addr, shift_data)
    log.info("  uploads done in %.1fs, exec_program starting...", time.monotonic() - t0)

    t1 = time.monotonic()
    cycles = transport.exec_program(program)
    log.info("  exec ok in %.1fs, %d cycles", time.monotonic() - t1, cycles)

    out_bytes = transport.read_mem(output_addr, output_size)
    full = np.frombuffer(out_bytes, dtype=np.int8).reshape(m, n)
    return full[:m_orig, :]


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
