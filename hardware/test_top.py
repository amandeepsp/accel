"""CFU top-level integration tests via valid/ready handshake.

Thorough quant tests live in epilogue/quant.py. These are smoke tests
that verify the CFU handshake wiring is correct.
"""

from amaranth.sim import Simulator

from top import Top


def to_signed32(val):
    if val >= (1 << 31):
        val -= 1 << 32
    return val


class TestCfuTop:
    """Smoke tests for the CFU valid/ready handshake."""

    async def _issue_cmd(self, ctx, dut, funct3, funct7, in0, in1, max_cycles=20):
        """Issue a command and wait for the response, returns the output."""
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
                return to_signed32(result)
            await ctx.tick()

        raise TimeoutError(f"CFU did not respond within {max_cycles} cycles")

    def test_srdhm_smoke(self):
        """SRDHM responds through the CFU handshake."""

        async def testbench(ctx):
            result = await self._issue_cmd(
                ctx, dut, funct3=3, funct7=0,
                in0=1000000, in1=1000000,
            )
            # Just verify we get a non-zero response — thorough tests in quant.py
            assert result != 0

        dut = Top()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_cfu_srdhm.vcd"):
            sim.run()

    def test_rdbpot_smoke(self):
        """RDBPOT responds through the CFU handshake."""

        async def testbench(ctx):
            result = await self._issue_cmd(
                ctx, dut, funct3=4, funct7=0,
                in0=100, in1=2,
            )
            assert result == 25  # 100 / 4

        dut = Top()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_cfu_rdbpot.vcd"):
            sim.run()

    def test_fallback_instruction(self):
        """Unused funct3 slots return in0 (fallback behavior)."""

        async def testbench(ctx):
            result = await self._issue_cmd(
                ctx, dut, funct3=7, funct7=0,
                in0=0xDEADBEEF, in1=0,
            )
            assert result == to_signed32(0xDEADBEEF)

        dut = Top()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_cfu_fallback.vcd"):
            sim.run()
