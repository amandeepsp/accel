#!/usr/bin/env python3
"""IR bytecode builder for the accel accelerator.

Exposes the same ProgramBuilder class as tools/e2e_gemm.py, plus helpers
for generating simple test programs (non-pipelined and pipelined GEMM).
"""

import struct
from dataclasses import dataclass
from typing import Optional

# IR Constants
KIR_MAGIC = 0x4B495200  # "KIR\0"
KIR_VERSION = 1
# Instruction opcodes
TILE_LOAD_ACT = 0x01
TILE_LOAD_WGT = 0x02
TILE_MMA = 0x03
TILE_STORE = 0x04
SET_EPILOGUE = 0x05
DONE = 0x06
# Tensor dtype
DTYPE_I8 = 0x0
DTYPE_I32 = 0x1


@dataclass(frozen=True)
class MemoryLayout:
    bias_addr: int
    mult_addr: int
    shift_addr: int
    weights_addr: int
    input_addr: int
    output_addr: int


class ProgramBuilder:
    """Build a Kernel IR program bytecode."""

    def __init__(self):
        self.tensors: list[tuple[int, int, int, int, int]] = []
        self.code = bytearray()
        self.instruction_count = 0

    def add_tensor(
        self, addr: int, dim0: int, dim1: int, stride: int, dtype: int = DTYPE_I8
    ) -> int:
        tensor_id = len(self.tensors)
        self.tensors.append((addr, dim0, dim1, stride, dtype))
        return tensor_id

    def emit_u8(self, val: int) -> None:
        self.code.append(val & 0xFF)

    def emit_u16_le(self, val: int) -> None:
        self.code.extend(struct.pack("<H", val))

    def emit_u32_le(self, val: int) -> None:
        self.code.extend(struct.pack("<I", val))

    def emit_i8(self, val: int) -> None:
        self.code.append(val & 0xFF)

    def tile_load_act(
        self, tensor_id: int, m_offset: int, k_offset: int, k_words: int
    ) -> None:
        self.emit_u8(TILE_LOAD_ACT)
        self.emit_u8(tensor_id)
        self.emit_u16_le(m_offset)
        self.emit_u16_le(k_offset)
        self.emit_u16_le(k_words)
        self.instruction_count += 1

    def tile_load_wgt(
        self, tensor_id: int, n_offset: int, k_offset: int, k_words: int
    ) -> None:
        self.emit_u8(TILE_LOAD_WGT)
        self.emit_u8(tensor_id)
        self.emit_u16_le(n_offset)
        self.emit_u16_le(k_offset)
        self.emit_u16_le(k_words)
        self.instruction_count += 1

    def tile_mma(self, first: bool, last: bool, k_count: int) -> None:
        self.emit_u8(TILE_MMA)
        flags = (1 if first else 0) | (2 if last else 0)
        self.emit_u8(flags)
        self.emit_u16_le(k_count)
        self.instruction_count += 1

    def tile_store(
        self, tensor_id: int, m_offset: int, n_offset: int, m_count: int, n_count: int
    ) -> None:
        self.emit_u8(TILE_STORE)
        self.emit_u8(tensor_id)
        self.emit_u16_le(m_offset)
        self.emit_u16_le(n_offset)
        self.emit_u8(m_count)
        self.emit_u8(n_count)
        self.instruction_count += 1

    def set_epilogue(
        self,
        bias_id: int,
        mult_id: int,
        shift_id: int,
        n_offset: int,
        n_count: int,
        output_offset: int,
        act_min: int,
        act_max: int,
    ) -> None:
        self.emit_u8(SET_EPILOGUE)
        self.emit_u8(bias_id)
        self.emit_u8(mult_id)
        self.emit_u8(shift_id)
        self.emit_u16_le(n_offset)
        self.emit_u16_le(n_count)
        self.emit_i8(output_offset)
        self.emit_i8(act_min)
        self.emit_i8(act_max)
        self.emit_u8(0)  # padding to 12 bytes
        self.instruction_count += 1

    def done(self) -> None:
        self.emit_u8(DONE)
        self.emit_u8(0)
        self.emit_u8(0)
        self.emit_u8(0)
        self.instruction_count += 1

    def build(self) -> bytes:
        program = bytearray()
        program.extend(struct.pack("<I", KIR_MAGIC))
        program.append(KIR_VERSION)
        program.append(len(self.tensors))
        program.extend(struct.pack("<H", self.instruction_count))
        for addr, dim0, dim1, stride, dtype in self.tensors:
            program.extend(struct.pack("<I", addr))
            program.extend(struct.pack("<HH", dim0, dim1))
            program.extend(struct.pack("<H", stride))
            program.append(dtype)
            program.append(0)  # flags
            program.extend(struct.pack("<I", 0))  # padding
        program.extend(self.code)
        return bytes(program)


def plan_memory(
    input_addr: int,
    weight_addr: int,
    output_addr: int,
    bias_addr: int,
    mult_addr: int,
    shift_addr: int,
) -> MemoryLayout:
    """Plan memory layout given explicit base addresses."""
    return MemoryLayout(
        bias_addr=bias_addr,
        mult_addr=mult_addr,
        shift_addr=shift_addr,
        weights_addr=weight_addr,
        input_addr=input_addr,
        output_addr=output_addr,
    )


def build_non_pipelined_gemm_program(
    layout: MemoryLayout,
    m: int,
    k: int,
    n: int,
    act_tensor_id: int,
    wgt_tensor_id: int,
    out_tensor_id: int,
    bias_id: int,
    mult_id: int,
    shift_id: int,
    tile: int = 8,
    k_tile: int = 8,
) -> bytes:
    """Build a non-pipelined GEMM program with software M and N tiling.

    This variant loads all data, waits for DMA, then computes. It does NOT
    overlap DMA with compute, so it tests the basic CFU compute path without
    DMA pipeline complexity.

    Supports software tiling for M and N (multiple m_base/n_base iterations).
    """
    builder = ProgramBuilder()

    # Add tensors (order matters for tensor IDs)
    # Input: [K, tile] column-major layout per M-tile, stride = K * tile bytes (size of one M-tile)
    builder.add_tensor(layout.input_addr, m, k, k * tile, DTYPE_I8)
    # Weights: row-major layout, stride = N bytes
    builder.add_tensor(layout.weights_addr, k, n, n, DTYPE_I8)
    # Output: row-major layout, stride = N bytes
    builder.add_tensor(layout.output_addr, m, n, n, DTYPE_I8)
    builder.add_tensor(layout.bias_addr, 1, n, n * 4, DTYPE_I32)
    builder.add_tensor(layout.mult_addr, 1, n, n * 4, DTYPE_I32)
    builder.add_tensor(layout.shift_addr, 1, n, n * 4, DTYPE_I32)

    # Software N and M tiling
    for n_base in range(0, n, tile):
        n_count = min(tile, n - n_base)
        for m_base in range(0, m, tile):
            m_count = min(tile, m - m_base)
            builder.set_epilogue(
                bias_id=bias_id,
                mult_id=mult_id,
                shift_id=shift_id,
                n_offset=n_base,
                n_count=n_count,
                output_offset=0,
                act_min=-128,
                act_max=127,
            )
            for k_base in range(0, k, k_tile):
                k_count = min(k_tile, k - k_base)
                # Input: [K, tile] layout per M-tile, stride = K*tile bytes between M-tiles
                # k_offset is byte offset within M-tile's K dimension: k_base * tile bytes
                builder.tile_load_act(
                    act_tensor_id,
                    m_offset=m_base,
                    k_offset=k_base * tile,
                    k_words=k_count,
                )
                # Weights: row-major, k_offset is K index, n_offset is column offset
                builder.tile_load_wgt(
                    wgt_tensor_id,
                    n_offset=n_base,
                    k_offset=k_base,
                    k_words=k_count,
                )
                builder.tile_mma(
                    first=(k_base == 0),
                    last=(k_base + k_count == k),
                    k_count=k_count,
                )
            builder.tile_store(
                out_tensor_id,
                m_offset=m_base,
                n_offset=n_base,
                m_count=m_count,
                n_count=n_count,
            )

    builder.done()
    return builder.build()


def build_pipelined_gemm_program(
    layout: MemoryLayout,
    m: int,
    k: int,
    n: int,
    tile: int,
    act_tensor_id: int,
    wgt_tensor_id: int,
    out_tensor_id: int,
    bias_id: int,
    mult_id: int,
    shift_id: int,
    cfu_word_bits: int = 64,
    cfu_store_depth_words: int = 512,
    k_tile: int | None = None,
) -> bytes:
    """Build a pipelined GEMM program with K-tiling and software M/N tiling.

    This variant overlaps DMA with compute: while one K-tile is being computed,
    the next K-tile's data is being DMA'd. It tests the full DMA pipeline.

    Supports software tiling for M and N (multiple m_base/n_base iterations).
    """
    if k_tile is None:
        dma_beats_per_line = cfu_word_bits // 32
        k_tile = cfu_store_depth_words // dma_beats_per_line

    builder = ProgramBuilder()

    # Add tensors (order matters for tensor IDs)
    # Input: [K, tile] column-major layout per M-tile, stride = K * tile bytes (size of one M-tile)
    builder.add_tensor(layout.input_addr, m, k, k * tile, DTYPE_I8)
    # Weights: row-major layout, stride = N bytes
    builder.add_tensor(layout.weights_addr, k, n, n, DTYPE_I8)
    # Output: row-major layout, stride = N bytes
    builder.add_tensor(layout.output_addr, m, n, n, DTYPE_I8)
    builder.add_tensor(layout.bias_addr, 1, n, n * 4, DTYPE_I32)
    builder.add_tensor(layout.mult_addr, 1, n, n * 4, DTYPE_I32)
    builder.add_tensor(layout.shift_addr, 1, n, n * 4, DTYPE_I32)

    # Software N and M tiling
    for n_base in range(0, n, tile):
        n_count = min(tile, n - n_base)
        for m_base in range(0, m, tile):
            m_count = min(tile, m - m_base)
            builder.set_epilogue(
                bias_id=bias_id,
                mult_id=mult_id,
                shift_id=shift_id,
                n_offset=n_base,
                n_count=n_count,
                output_offset=0,
                act_min=-128,
                act_max=127,
            )
            for k_base in range(0, k, k_tile):
                k_count = min(k_tile, k - k_base)
                # Input: [K, tile] layout per M-tile, stride = K*tile bytes between M-tiles
                # k_offset is byte offset within M-tile's K dimension: k_base * tile bytes
                builder.tile_load_act(
                    act_tensor_id,
                    m_offset=m_base,
                    k_offset=k_base * tile,
                    k_words=k_count,
                )
                # Weights: row-major, k_offset is K index, n_offset is column offset
                builder.tile_load_wgt(
                    wgt_tensor_id,
                    n_offset=n_base,
                    k_offset=k_base,
                    k_words=k_count,
                )
                builder.tile_mma(
                    first=(k_base == 0),
                    last=(k_base + k_count == k),
                    k_count=k_count,
                )
            builder.tile_store(
                out_tensor_id,
                m_offset=m_base,
                n_offset=n_base,
                m_count=m_count,
                n_count=n_count,
            )

    builder.done()
    return builder.build()
