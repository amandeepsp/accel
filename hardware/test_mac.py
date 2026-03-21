"""SimdMac4 simulation tests."""

from amaranth.sim import Simulator

from conftest import pack_bytes, to_signed32
from mac import SimdMac4


class TestSimdMac4:
    def test_basic_mac(self):
        """[1,2,3,4] * [5,6,7,8] with offset=128."""

        async def testbench(ctx):
            ctx.set(dut.input_offset, 128)
            ctx.set(dut.reset_acc, 1)
            await ctx.tick()
            ctx.set(dut.reset_acc, 0)
            await ctx.tick()

            ctx.set(dut.in0, pack_bytes(1, 2, 3, 4))
            ctx.set(dut.in1, pack_bytes(5, 6, 7, 8))
            ctx.set(dut.start, 1)
            await ctx.tick()
            ctx.set(dut.start, 0)
            await ctx.tick()

            acc = to_signed32(ctx.get(dut.accumulator))
            # (1+128)*5 + (2+128)*6 + (3+128)*7 + (4+128)*8
            # = 645 + 780 + 917 + 1056 = 3398
            assert acc == 3398, f"expected 3398, got {acc}"

        dut = SimdMac4()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("test_mac4.vcd"):
            sim.run()

    def test_accumulate_two_rounds(self):
        """Two consecutive MACs accumulate."""

        async def testbench(ctx):
            ctx.set(dut.input_offset, 128)
            ctx.set(dut.reset_acc, 1)
            await ctx.tick()
            ctx.set(dut.reset_acc, 0)
            await ctx.tick()

            # Round 1: [1,1,1,1] * [1,1,1,1] = 4 * 129 = 516
            ctx.set(dut.in0, pack_bytes(1, 1, 1, 1))
            ctx.set(dut.in1, pack_bytes(1, 1, 1, 1))
            ctx.set(dut.start, 1)
            await ctx.tick()

            # Round 2: same again, acc should be 516 + 516 = 1032
            await ctx.tick()
            ctx.set(dut.start, 0)
            await ctx.tick()

            acc = to_signed32(ctx.get(dut.accumulator))
            assert acc == 1032, f"expected 1032, got {acc}"

        dut = SimdMac4()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("test_mac4_accum.vcd"):
            sim.run()

    def test_zero_inputs(self):
        """All zero inputs with offset=0 produces 0."""

        async def testbench(ctx):
            ctx.set(dut.input_offset, 0)
            ctx.set(dut.reset_acc, 1)
            await ctx.tick()
            ctx.set(dut.reset_acc, 0)
            await ctx.tick()

            ctx.set(dut.in0, 0)
            ctx.set(dut.in1, 0)
            ctx.set(dut.start, 1)
            await ctx.tick()
            ctx.set(dut.start, 0)
            await ctx.tick()

            acc = to_signed32(ctx.get(dut.accumulator))
            assert acc == 0, f"expected 0, got {acc}"

        dut = SimdMac4()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("test_mac4_zero.vcd"):
            sim.run()

    def test_negative_values(self):
        """Negative inputs work correctly."""

        async def testbench(ctx):
            ctx.set(dut.input_offset, 0)
            ctx.set(dut.reset_acc, 1)
            await ctx.tick()
            ctx.set(dut.reset_acc, 0)
            await ctx.tick()

            # [-1, -2, 0, 0] * [3, 4, 0, 0] = (-1)*3 + (-2)*4 = -11
            ctx.set(dut.in0, pack_bytes(-1, -2, 0, 0))
            ctx.set(dut.in1, pack_bytes(3, 4, 0, 0))
            ctx.set(dut.start, 1)
            await ctx.tick()
            ctx.set(dut.start, 0)
            await ctx.tick()

            acc = to_signed32(ctx.get(dut.accumulator))
            assert acc == -11, f"expected -11, got {acc}"

        dut = SimdMac4()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("test_mac4_neg.vcd"):
            sim.run()

    def test_reset_clears_accumulator(self):
        """Reset brings accumulator back to 0 after accumulation."""

        async def testbench(ctx):
            ctx.set(dut.input_offset, 0)
            ctx.set(dut.reset_acc, 1)
            await ctx.tick()
            ctx.set(dut.reset_acc, 0)
            await ctx.tick()

            # Accumulate something
            ctx.set(dut.in0, pack_bytes(10, 10, 10, 10))
            ctx.set(dut.in1, pack_bytes(1, 1, 1, 1))
            ctx.set(dut.start, 1)
            await ctx.tick()
            ctx.set(dut.start, 0)
            await ctx.tick()

            acc = to_signed32(ctx.get(dut.accumulator))
            assert acc == 40, f"expected 40, got {acc}"

            # Reset
            ctx.set(dut.reset_acc, 1)
            await ctx.tick()
            ctx.set(dut.reset_acc, 0)
            await ctx.tick()

            acc = to_signed32(ctx.get(dut.accumulator))
            assert acc == 0, f"expected 0 after reset, got {acc}"

        dut = SimdMac4()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("test_mac4_reset.vcd"):
            sim.run()
