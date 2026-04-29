"""Test multi-tile (K-tiling) behavior in Amaranth simulation."""

import numpy as np
from amaranth.sim import Simulator

from hardware.testing import cfu_op, dma_fill, pack_int8, to_signed8, write_per_channel_params
from hardware.top import Top, TopConfig
from shared.reference import INT32_MAX, INT32_MIN, ref_epilogue


def test_8x8_two_tiles():
    """8x8 GEMM with K=16 split into two tiles of K=8."""
    ROWS, COLS, K = 8, 8, 16
    K_TILE = 8

    # Deterministic all-1s for easy verification
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

        # --- Epilogue params (must be set before last tile) ---
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
        # compute_wait
        await cfu_op(ctx, dut, funct3=1, funct7=0, in0=0, in1=0)

        # --- Tile 1: K=8..15 ---
        await dma_fill(ctx, dut.dma_act, act_words[K_TILE:])
        await dma_fill(ctx, dut.dma_wgt, wgt_words[K_TILE:])

        # compute_start first=False, last=True
        flags = 0 | (1 << 1)
        await cfu_op(ctx, dut, funct3=0, funct7=0, in0=flags, in1=K_TILE)
        # compute_wait
        await cfu_op(ctx, dut, funct3=1, funct7=0, in0=0, in1=0)

        print(f"After tile1 compute_wait: state_debug={ctx.get(dut.seq.state_debug)}, busy={ctx.get(dut.seq.busy_debug)}")

        # --- Read results ---
        for i in range(num_ch):
            got = await cfu_op(ctx, dut, funct3=4, funct7=0, in0=i, in1=0)
            got = to_signed8(got)
            assert got == expected[i], f"result[{i}]: got {got}, expected {expected[i]}"

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("waves/test_multi_tile.vcd"):
        sim.run()
    print("PASS")


if __name__ == "__main__":
    test_8x8_two_tiles()
