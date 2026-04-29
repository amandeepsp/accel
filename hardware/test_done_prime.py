"""Verify DONE→PRIME transition and computeWait behavior."""

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


async def sample_seq(ctx, dut, label=""):
    state = ctx.get(dut.seq.state_debug)
    busy = ctx.get(dut.seq.busy_debug)
    done = ctx.get(dut.seq.done)
    start = ctx.get(dut.seq.start)
    first = ctx.get(dut.seq.first)
    first_latch = ctx.get(dut.seq.first_latch_debug)
    psum_load = ctx.get(dut.seq.arr_psum_load)
    print(
        f"{label:20s}  state={state} busy={busy} done={done} start={start} "
        f"first={first} first_latch={first_latch} psum_load={psum_load}"
    )
    return state, busy, done, start, first, first_latch, psum_load


def test_back_to_back_tiles():
    """Top-level test: back-to-back tiles, probe first_latch and state transitions."""
    ROWS, COLS, K = 2, 2, 2
    K_TILE = 2
    num_ch = ROWS * COLS

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

    dut = Top(TopConfig(rows=ROWS, cols=COLS, store_depth=16))

    async def testbench(ctx):
        # Setup
        biases = [BIAS] * num_ch
        mults = [MULT] * num_ch
        shifts = [SHIFT] * num_ch
        await write_per_channel_params(ctx, dut, biases, mults, shifts)
        await cfu_op(ctx, dut, funct3=3, funct7=0, in0=0, in1=OFFSET)
        await cfu_op(ctx, dut, funct3=3, funct7=1, in0=0, in1=ACT_MIN & 0xFF)
        await cfu_op(ctx, dut, funct3=3, funct7=2, in0=0, in1=ACT_MAX & 0xFF)

        # ------------------------------------------------------------------
        # Tile 0: first=1, last=0  (middle tile of a K-tile sequence)
        # ------------------------------------------------------------------
        await dma_fill(ctx, dut.dma_act, act_words[:K_TILE])
        await dma_fill(ctx, dut.dma_wgt, wgt_words[:K_TILE])

        flags = 1 | (0 << 1)
        await cfu_op(ctx, dut, funct3=0, funct7=0, in0=flags, in1=K_TILE)
        # cfu_op returns at the beginning of the cycle where the sequencer has
        # just entered PRIME (seq_start went high on the previous cycle).
        await sample_seq(ctx, dut, "tile0 PRIME")

        await cfu_op(ctx, dut, funct3=1, funct7=0, in0=0, in1=0)
        state0, busy0, done0, start0, first0, latch0, psum0 = await sample_seq(ctx, dut, "after tile0 wait")

        # After a non-last tile the sequencer reaches DONE then falls back to IDLE
        # because the one-cycle start pulse has already passed.
        assert state0 == 0, f"Expected IDLE (0) after tile0, got {state0}"
        # first_latch retains its old value until the next start; that is expected.

        # ------------------------------------------------------------------
        # Tile 1: first=0, last=1  (final tile)
        # ------------------------------------------------------------------
        await dma_fill(ctx, dut.dma_act, act_words[:K_TILE])
        await dma_fill(ctx, dut.dma_wgt, wgt_words[:K_TILE])

        flags = 0 | (1 << 1)
        await cfu_op(ctx, dut, funct3=0, funct7=0, in0=flags, in1=K_TILE)
        state_prime1, _, _, _, _, latch_prime1, psum_prime1 = await sample_seq(ctx, dut, "tile1 PRIME")

        assert state_prime1 == 1, f"Expected PRIME (1) after tile1 start, got {state_prime1}"
        assert latch_prime1 == 0, f"first_latch should be 0 for non-first tile, got {latch_prime1}"
        assert psum_prime1 == 0, f"psum_load should be 0 when first=0, got {psum_prime1}"

        await cfu_op(ctx, dut, funct3=1, funct7=0, in0=0, in1=0)
        state1, busy1, done1, start1, first1, latch1, psum1 = await sample_seq(ctx, dut, "after tile1 wait")

        assert state1 == 0, f"Expected IDLE (0) after tile1, got {state1}"

        # ------------------------------------------------------------------
        # Wait arbitrary cycles, then start a *new* program with first=1.
        # ------------------------------------------------------------------
        for _ in range(3):
            await ctx.tick()
        await sample_seq(ctx, dut, "before new prog")

        await dma_fill(ctx, dut.dma_act, act_words[:K_TILE])
        await dma_fill(ctx, dut.dma_wgt, wgt_words[:K_TILE])

        flags = 1 | (1 << 1)
        await cfu_op(ctx, dut, funct3=0, funct7=0, in0=flags, in1=K_TILE)
        state_prime2, _, _, _, _, latch_prime2, psum_prime2 = await sample_seq(ctx, dut, "new prog PRIME")

        assert state_prime2 == 1, f"Expected PRIME (1) for new program, got {state_prime2}"
        assert latch_prime2 == 1, f"first_latch should be 1 for first tile of new program, got {latch_prime2}"
        assert psum_prime2 == 1, f"psum_load should be 1 when first=1, got {psum_prime2}"

        await cfu_op(ctx, dut, funct3=1, funct7=0, in0=0, in1=0)
        state2, busy2, done2, start2, first2, latch2, psum2 = await sample_seq(ctx, dut, "after new wait")

        assert state2 == 0, f"Expected IDLE (0) after new program, got {state2}"

        # ------------------------------------------------------------------
        # Verify results are correct (catches accumulator-reset bugs)
        # ------------------------------------------------------------------
        for i in range(num_ch):
            got = await cfu_op(ctx, dut, funct3=4, funct7=0, in0=i, in1=0)
            got = to_signed8(got)
            assert got == expected[i], f"result[{i}]: got {got}, expected {expected[i]}"

        print("Top-level back-to-back test PASS")

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("waves/test_done_prime_top.vcd"):
        sim.run()


def test_sequencer_done_prime_direct():
    """Directly drive OSSequencer to hit DONE→PRIME on the exact same cycle."""
    from hardware.control.os_sequencer import OSSequencer

    seq = OSSequencer(rows=2, cols=2, scratchpad_depth=8)

    async def testbench(ctx):
        async def wait_state(target, max_cycles=200):
            mapping = {
                0: "IDLE",
                1: "PRIME",
                2: "FEED",
                3: "FLUSH",
                4: "EPILOGUE",
                5: "EPILOGUE_WAIT",
                6: "DONE",
            }
            for _ in range(max_cycles):
                st = ctx.get(seq.state_debug)
                if st == target:
                    return st
                await ctx.tick()
            raise TimeoutError(f"Timeout waiting for state {target} ({mapping.get(target, '?')})")

        # Idle
        await wait_state(0)

        # ------------------------------------------------------------------
        # Tile 0: first=1, last=1, k_count=2
        # ------------------------------------------------------------------
        # Hold first/last/k_count for the whole tile (mirrors ComputeStartInstruction regs)
        ctx.set(seq.start, 1)
        ctx.set(seq.first, 1)
        ctx.set(seq.last, 1)
        ctx.set(seq.k_count, 2)
        await ctx.tick()
        ctx.set(seq.start, 0)
        # keep first/last/k_count asserted

        await wait_state(1)  # PRIME
        assert ctx.get(seq.first_latch_debug) == 1, "first_latch should be 1 in PRIME"
        assert ctx.get(seq.arr_psum_load) == 1, "psum_load should be 1 in PRIME when first=1"

        # Log transitions until EPILOGUE
        for _ in range(20):
            st = ctx.get(seq.state_debug)
            if st == 4:
                break
            await ctx.tick()
        else:
            raise TimeoutError("Did not reach EPILOGUE")

        await wait_state(5)  # EPILOGUE_WAIT

        # epi_done is provided by Epilogue one cycle after last_in.
        # In this direct test we drive it manually.
        ctx.set(seq.epi_done, 1)
        await ctx.tick()
        ctx.set(seq.epi_done, 0)

        # Now in DONE (cycle N)
        st = await wait_state(6)
        assert ctx.get(seq.done) == 1

        # Deassert tile0 config before next tile
        ctx.set(seq.first, 0)
        ctx.set(seq.last, 0)

        # ------------------------------------------------------------------
        # Assert start on the exact same cycle DONE is entered (still in DONE)
        # with first=0 to test middle-tile restart.
        # ------------------------------------------------------------------
        ctx.set(seq.start, 1)
        ctx.set(seq.first, 0)
        await ctx.tick()
        ctx.set(seq.start, 0)
        ctx.set(seq.first, 0)

        # Next cycle must be PRIME with first_latch = 0
        assert ctx.get(seq.state_debug) == 1, f"Expected PRIME, got {ctx.get(seq.state_debug)}"
        assert ctx.get(seq.first_latch_debug) == 0, f"Expected first_latch=0, got {ctx.get(seq.first_latch_debug)}"
        assert ctx.get(seq.arr_psum_load) == 0, "psum_load should be 0 when first=0"

        # Continue to FEED, FLUSH, then DONE (because last was not set this time)
        await wait_state(2)
        await wait_state(3)
        await wait_state(6)

        # Deassert start -> IDLE
        await ctx.tick()
        assert ctx.get(seq.state_debug) == 0, f"Expected IDLE, got {ctx.get(seq.state_debug)}"

        # ------------------------------------------------------------------
        # New program start from IDLE with first=1
        # ------------------------------------------------------------------
        ctx.set(seq.start, 1)
        ctx.set(seq.first, 1)
        await ctx.tick()
        ctx.set(seq.start, 0)
        ctx.set(seq.first, 0)

        assert ctx.get(seq.state_debug) == 1
        assert ctx.get(seq.first_latch_debug) == 1

        print("Direct sequencer test PASS")

    sim = Simulator(seq)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    with sim.write_vcd("waves/test_done_prime_direct.vcd"):
        sim.run()


if __name__ == "__main__":
    test_back_to_back_tiles()
    test_sequencer_done_prime_direct()
