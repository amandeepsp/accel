"""SRDHM and RoundingDivideByPOT simulation tests."""

import pytest
from amaranth.sim import Simulator

from conftest import INT32_MIN, INT32_MAX, ref_srdhm, ref_rdbpot, to_signed32
from quant import SRDHM


class TestSRDHM:
    def test_saturation(self):
        """INT32_MIN * INT32_MIN saturates to INT32_MAX."""

        async def testbench(ctx):
            ctx.set(dut.a, INT32_MIN)
            ctx.set(dut.b, INT32_MIN)
            ctx.set(dut.start, 1)
            await ctx.tick()
            # Saturation path: done=1 is latched on this edge
            assert ctx.get(dut.done) == 1
            result = to_signed32(ctx.get(dut.out))
            assert result == INT32_MAX, f"expected {INT32_MAX}, got {result}"

        dut = SRDHM()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("test_srdhm_sat.vcd"):
            sim.run()

    @pytest.mark.parametrize(
        "a, b",
        [
            (1000000, 1000000),
            (-500000, 300000),
            (INT32_MAX, 1),
            (1, INT32_MAX),
            (INT32_MIN + 1, INT32_MIN + 1),
            (0, 12345),
            (123456789, -987654321),
        ],
    )
    def test_reference(self, a, b):
        """SRDHM matches Python reference for various inputs."""
        expected = ref_srdhm(a, b)

        async def testbench(ctx):
            ctx.set(dut.a, a)
            ctx.set(dut.b, b)
            ctx.set(dut.start, 1)
            await ctx.tick()
            ctx.set(dut.start, 0)
            # Normal path: 2 cycles (stage0 → multiply, stage1 → extract+done)
            await ctx.tick()
            assert ctx.get(dut.done) == 1
            result = to_signed32(ctx.get(dut.out))
            assert result == expected, f"srdhm({a}, {b}): expected {expected}, got {result}"

        dut = SRDHM()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("test_srdhm.vcd"):
            sim.run()

    def test_zero(self):
        """SRDHM with zero input gives zero."""

        async def testbench(ctx):
            ctx.set(dut.a, 0)
            ctx.set(dut.b, 12345)
            ctx.set(dut.start, 1)
            await ctx.tick()
            ctx.set(dut.start, 0)
            # Normal path: 2 cycles
            await ctx.tick()
            assert ctx.get(dut.done) == 1
            result = to_signed32(ctx.get(dut.out))
            assert result == 0, f"expected 0, got {result}"

        dut = SRDHM()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("test_srdhm_zero.vcd"):
            sim.run()


class TestRDBPOT:
    """RoundingDividebyPOT is purely combinational — no clock domain.

    The Python reference is tested directly here. The hardware is tested
    through the full CFU handshake in test_top.py::TestCfuTop.test_rdbpot_via_cfu.
    """

    @pytest.mark.parametrize(
        "x, exponent, expected",
        [
            (100, 2, 25),       # 100 / 4 = 25
            (101, 2, 25),       # 101 / 4 = 25.25 → 25
            (102, 2, 26),       # 102 / 4 = 25.5  → 26
            (103, 2, 26),       # 103 / 4 = 25.75 → 26
            (-100, 2, -25),     # -100 / 4 = -25
            (-101, 2, -25),     # rounds toward zero
            (-102, 2, -26),     # -102 / 4 = -25.5 → -26 (negative midpoint rounds away from zero)
            (-103, 2, -26),     # -103 / 4 = -25.75 → -26
            (256, 8, 1),
            (255, 8, 1),
            (127, 8, 0),
            (0, 5, 0),
            (1, 1, 1),         # 1 / 2 = 0.5 → 1
            (-1, 1, -1),       # -1 / 2 = -0.5 → -1 (rounds away from zero)
        ],
    )
    def test_reference(self, x, exponent, expected):
        """Python reference produces correct rounding."""
        result = ref_rdbpot(x, exponent)
        assert result == expected, f"rdbpot({x}, {exponent}): expected {expected}, got {result}"
