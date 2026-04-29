"""Test pipelined multi-tile (K-tiling) behavior in Amaranth simulation."""

import numpy as np
from amaranth.sim import Simulator

from hardware.top import Top, TopConfig

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


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
    return int((x >> exponent) + (1 if remainder > threshold else 0))


def ref_epilogue(acc, bias, multiplier, shift, offset, act_min, act_max):
    x = acc + bias
    x = ref_srdhm(x, multiplier)
    x = ref_rdbpot(x, shift)
    x += offset
    return max(act_min, min(act_max, x))


def pack_int8(vals):
    word = 0
    for i, v in enumerate(vals):
        word |= (v & 0xFF) << (8 * i)
    return word


def to_signed8(val):
    val = val & 0xFF
    return val - 256 if val >= 128 else val


async def dma_fill(ctx, dma_port, words):
    for addr, word in enumerate(words):
        ctx.set(dma_port.addr, addr)
        ctx.set(dma_port.data, word)
        ctx.set(dma_port.en, 1)
        await ctx.tick()
    ctx.set(dma_port.en, 0)
    await ctx.tick()


async def cfu_op(ctx, dut, funct3, funct7, in0, in1, max_cycles=5000):
    ctx.set(dut.cmd_valid, 1)
    ctx.set(dut.cmd_function_id, {"funct3": funct3, "funct7": funct7})
    ctx.set(dut.cmd_in0, in0)
    ctx.set(dut.cmd_in1, in1)
    ctx.set(dut.rsp_ready, 1)
    await ctx.tick()

    for _ in range(max_cycles):
        if ctx.get(dut.rsp_valid):
            result = ctx.get(dut.rsp_out)
            ctx.set(dut.cmd_valid, 0)
            await ctx.tick()
            return result
        await ctx.tick()
    raise TimeoutError(f"CFU did not respond within {max_cycles} cycles")


async def write_per_channel_params(ctx, dut, biases, mults, shifts):
    for ch in range(len(biases)):
        await cfu_op(ctx, dut, funct3=2, funct7=0, in0=ch, in1=biases[ch])
        await cfu_op(ctx, dut, funct3=2, funct7=1, in0=ch, in1=mults[ch])
        await cfu_op(ctx, dut, funct3=2, funct7=2, in0=ch, in1=shifts[ch])


def test_8x8_two_tiles_pipelined():
    """8x8 GEMM with K=16 split into two tiles of K=8, pipelined DMA."""
    ROWS, COLS, K = 8, 8, 16
    K_TILE = 8

    A = np.ones((ROWS, K), dtype=np.int8)
    B = np.ones((K, COLS), dtype=np.int8)
    C = A.astype(np.int32) @ B.astype(np.int32)

    MULT = 1 << 30
    SHIFT = 0
    BIAS = 0
    OFFSET = 0
    ACT_MIN, ACT_MAX = -128, 127

    expected = []
    for r in range(ROWS):
        for c in range(COLS):
            expected.append(ref_epilogue(int(C[r, c]), BIAS, MULT, SHIFT, OFFSET, ACT_MIN, ACT_MAX))

    act_words = [pack_int8([int(A[r, k]) for r in range(ROWS)]) for k in range(K)]
    wgt_words = [pack_int8([int(B[k, c]) for c in range(COLS)]) for k in range(K)]

    dut = Top(TopConfig(rows=ROWS, cols=COLS))

    async def testbench(ctx):
        num_ch = ROWS * COLS

        # --- Epilogue params ---
        biases = [BIAS] * num_ch
        mults = [MULT] * num_ch
        shifts = [SHIFT] * num_ch
        await write_per_channel_params(ctx, dut, biases, mults, shifts)

        # --- Global config ---
        await cfu_op(ctx, dut, funct3=3, funct7=0, in0=0, in1=OFFSET)
        await cfu_op(ctx, dut, funct3=3, funct7=1, in0=0, in1=ACT_MIN & 0xFF)
        await cfu_op(ctx, dut, funct3=3, funct7=2, in0=0, in1=ACT_MAX & 0xFF)

        # --- Tile 0: K=0..7 ---
        await dma_fill(ctx, dut.dma_act, act_words[:K_TILE])
        await dma_fill(ctx, dut.dma_wgt, wgt_words[:K_TILE])

        # compute_start first=True, last=False
        flags = 1 | (0 << 1)
        await cfu_op(ctx, dut, funct3=0, funct7=0, in0=flags, in1=K_TILE)

        # --- OVERLAPPED: Tile 1 DMA while Tile 0 computes ---
        await dma_fill(ctx, dut.dma_act, act_words[K_TILE:])
        await dma_fill(ctx, dut.dma_wgt, wgt_words[K_TILE:])

        # compute_wait for tile 0
        await cfu_op(ctx, dut, funct3=1, funct7=0, in0=0, in1=0)

        # --- Tile 1: compute ---
        flags = 0 | (1 << 1)
        await cfu_op(ctx, dut, funct3=0, funct7=0, in0=flags, in1=K_TILE)
        # compute_wait for tile 1
        await cfu_op(ctx, dut, funct3=1, funct7=0, in0=0, in1=0)

        # --- Read results ---
        for i in range(num_ch):
            got = await cfu_op(ctx, dut, funct3=4, funct7=0, in0=i, in1=0)
            got = to_signed8(got)
            assert got == expected[i], f"result[{i}]: got {got}, expected {expected[i]}"

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("waves/test_multi_tile_pipelined.vcd"):
        sim.run()
    print("PASS")


if __name__ == "__main__":
    test_8x8_two_tiles_pipelined()
