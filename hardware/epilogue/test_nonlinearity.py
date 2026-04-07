import pytest

from nonlinearity import ReLU, ReLU6


class TestReLU:
    @pytest.mark.parametrize(
        "x, expected",
        [
            (10, 10),
            (0, 0),
            (-1, 0),
            (-1000, 0),
            (127, 127),
        ],
    )
    def test_values(self, x, expected):
        """ReLU is combinational — test via elaborate + Python eval."""
        from amaranth.sim import Simulator

        async def testbench(ctx):
            ctx.set(dut.x, x)
            await ctx.delay(1e-7)
            assert ctx.get(dut.out) == expected, \
                f"relu({x}) = {ctx.get(dut.out)}, expected {expected}"

        dut = ReLU()
        sim = Simulator(dut)
        sim.add_testbench(testbench)
        sim.run()


class TestReLU6:
    @pytest.mark.parametrize(
        "x, expected",
        [
            (3, 3),
            (6, 6),
            (7, 6),
            (100, 6),
            (0, 0),
            (-1, 0),
            (-100, 0),
        ],
    )
    def test_values(self, x, expected):
        from amaranth.sim import Simulator

        async def testbench(ctx):
            ctx.set(dut.x, x)
            await ctx.delay(1e-7)
            assert ctx.get(dut.out) == expected, \
                f"relu6({x}) = {ctx.get(dut.out)}, expected {expected}"

        dut = ReLU6()
        sim = Simulator(dut)
        sim.add_testbench(testbench)
        sim.run()

    def test_custom_clamp(self):
        from amaranth.sim import Simulator

        async def testbench(ctx):
            ctx.set(dut.x, 200)
            await ctx.delay(1e-7)
            assert ctx.get(dut.out) == 128

            ctx.set(dut.x, 50)
            await ctx.delay(1e-7)
            assert ctx.get(dut.out) == 50

            ctx.set(dut.x, -10)
            await ctx.delay(1e-7)
            assert ctx.get(dut.out) == 0

        dut = ReLU6(clamp=128)
        sim = Simulator(dut)
        sim.add_testbench(testbench)
        sim.run()
