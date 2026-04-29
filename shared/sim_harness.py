"""Shared GEMM simulation harness for Verilator/end-to-end tests."""

import logging
import time

import numpy as np

from shared.ir import build_gemm_program, patch_epilogue, plan_memory
from shared.layout import align_up, pack_input_tiles, pack_weight_rows

log = logging.getLogger("accel.sim_harness")

MEM_ALIGN = 32


def run_gemm_on_sim(
    transport,
    lhs_orig,
    rhs,
    *,
    bias=None,
    multiplier=None,
    shift=None,
    output_offset=0,
    activation_min=-128,
    activation_max=127,
    tile=8,
    cfu_word_bits=64,
    cfu_store_depth_words=512,
    tensor_pool_base=0x40010100,
):
    """Execute a full GEMM (any M,N) on the sim via the transport."""
    m_orig, k = lhs_orig.shape
    _k2, n = rhs.shape
    assert k == _k2

    if bias is None:
        bias = np.zeros(n, dtype=np.int32)
    if multiplier is None:
        multiplier = np.ones(n, dtype=np.int32) * (1 << 30)
    if shift is None:
        shift = np.zeros(n, dtype=np.int32)

    # Zero-pad M to tile multiple (hardware requires full-width DMA rows)
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

    layout = plan_memory(
        input_addr, weight_addr, output_addr, bias_addr, mult_addr, shift_addr
    )
    program = build_gemm_program(
        layout,
        m,
        k,
        n,
        tile,
        act_tensor_id=0,
        wgt_tensor_id=1,
        out_tensor_id=2,
        bias_id=3,
        mult_id=4,
        shift_id=5,
        cfu_word_bits=cfu_word_bits,
        cfu_store_depth_words=cfu_store_depth_words,
    )

    program = patch_epilogue(
        program,
        output_offset=output_offset,
        activation_min=activation_min,
        activation_max=activation_max,
    )

    log.info(
        "  GEMM %dx%dx%d (padded M=%d): uploading %d input + %d weight + %d epi bytes, "
        "program=%d bytes",
        m_orig,
        k,
        n,
        m,
        len(input_data),
        len(weight_data),
        len(bias_data) + len(mult_data) + len(shift_data),
        len(program),
    )
    t0 = time.monotonic()
    transport.write_mem(input_addr, input_data)
    transport.write_mem(weight_addr, weight_data)
    transport.write_mem(bias_addr, bias_data)
    transport.write_mem(mult_addr, mult_data)
    transport.write_mem(shift_addr, shift_data)
    log.info(
        "  GEMM %dx%dx%d: uploads done in %.1fs, exec_program starting...",
        m_orig,
        k,
        n,
        time.monotonic() - t0,
    )

    t1 = time.monotonic()
    cycles = transport.exec_program(program)
    log.info(
        "  GEMM %dx%dx%d: exec ok in %.1fs, %d cycles",
        m_orig,
        k,
        n,
        time.monotonic() - t1,
        cycles,
    )

    out_bytes = transport.read_mem(output_addr, output_size)
    full = np.frombuffer(out_bytes, dtype=np.int8).reshape(m, n)
    return full[:m_orig, :]
