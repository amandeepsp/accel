"""Full CFU top-level integration tests via valid/ready handshake."""

from amaranth.sim import Simulator

from conftest import (
    INT32_MIN, INT32_MAX,
    pack_bytes, ref_srdhm, ref_rdbpot, to_signed32,
)
from top import Top


class TestCfuTop:
    """Tests that drive the full CFU valid/ready handshake."""

    async def _issue_cmd(self, ctx, dut, funct3, funct7, in0, in1, max_cycles=20):
        """Issue a command and wait for the response, returns the output."""
        function_id = (funct7 << 3) | funct3

        ctx.set(dut.cmd_valid, 1)
        ctx.set(dut.cmd_function_id, function_id)
        ctx.set(dut.cmd_in0, in0)
        ctx.set(dut.cmd_in1, in1)
        ctx.set(dut.rsp_ready, 1)
        await ctx.tick()

        # Wait for rsp_valid
        for _ in range(max_cycles):
            if ctx.get(dut.rsp_valid):
                result = ctx.get(dut.rsp_out)
                ctx.set(dut.cmd_valid, 0)
                await ctx.tick()
                return to_signed32(result)
            await ctx.tick()

        raise TimeoutError(f"CFU did not respond within {max_cycles} cycles")

    def test_mac4_via_cfu(self):
        """MAC4 through the full CFU handshake."""

        async def testbench(ctx):
            # Set input offset to 128 via WriteRegs (funct3=2, in0=0 selects offset)
            await self._issue_cmd(ctx, dut, funct3=2, funct7=0, in0=0, in1=128)

            # Reset accumulator via WriteRegs (funct3=2, in0=1 selects reset)
            await self._issue_cmd(ctx, dut, funct3=2, funct7=0, in0=1, in1=0)

            # MAC4 (funct3=0): [1,0,0,0] * [1,0,0,0]
            # = (1+128)*1 = 129
            in0 = pack_bytes(1, 0, 0, 0)
            in1 = pack_bytes(1, 0, 0, 0)
            await self._issue_cmd(ctx, dut, funct3=0, funct7=0, in0=in0, in1=in1)

            # Read accumulator via ReadRegs (funct3=1, in0=1 selects accumulator)
            result = await self._issue_cmd(ctx, dut, funct3=1, funct7=0, in0=1, in1=0)
            assert result == 129, f"expected 129, got {result}"

        dut = Top()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("test_cfu_mac4.vcd"):
            sim.run()

    def test_srdhm_via_cfu(self):
        """SRDHM (multi-cycle) through the full CFU handshake."""
        a, b = 1000000, 1000000
        expected = ref_srdhm(a, b)

        async def testbench(ctx):
            result = await self._issue_cmd(
                ctx, dut,
                funct3=3, funct7=0,
                in0=a & 0xFFFFFFFF, in1=b & 0xFFFFFFFF,
            )
            assert result == expected, f"srdhm({a},{b}): expected {expected}, got {result}"

        dut = Top()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("test_cfu_srdhm.vcd"):
            sim.run()

    def test_rdbpot_via_cfu(self):
        """RDBPOT (single-cycle) through the full CFU handshake."""
        x, exponent = 100, 2
        expected = ref_rdbpot(x, exponent)

        async def testbench(ctx):
            result = await self._issue_cmd(
                ctx, dut,
                funct3=4, funct7=0,
                in0=x & 0xFFFFFFFF, in1=exponent,
            )
            assert result == expected, f"rdbpot({x},{exponent}): expected {expected}, got {result}"

        dut = Top()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("test_cfu_rdbpot.vcd"):
            sim.run()

    def test_srdhm_saturation_via_cfu(self):
        """SRDHM saturation case through the full CFU handshake."""

        async def testbench(ctx):
            result = await self._issue_cmd(
                ctx, dut,
                funct3=3, funct7=0,
                in0=INT32_MIN & 0xFFFFFFFF, in1=INT32_MIN & 0xFFFFFFFF,
            )
            assert result == INT32_MAX, f"expected {INT32_MAX}, got {result}"

        dut = Top()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("test_cfu_srdhm_sat.vcd"):
            sim.run()

    def test_full_requant_pipeline(self):
        """MAC4 → SRDHM → RDBPOT: the full quantized conv output transform."""
        multiplier = 1073741824  # 0.5 in Q31 fixed-point
        shift = 2

        async def testbench(ctx):
            # Set offset=0, reset acc
            await self._issue_cmd(ctx, dut, funct3=2, funct7=0, in0=0, in1=0)
            await self._issue_cmd(ctx, dut, funct3=2, funct7=0, in0=1, in1=0)

            # MAC4: [10,20,30,40] * [1,1,1,1] = 10+20+30+40 = 100
            in0 = pack_bytes(10, 20, 30, 40)
            in1 = pack_bytes(1, 1, 1, 1)
            await self._issue_cmd(ctx, dut, funct3=0, funct7=0, in0=in0, in1=in1)

            # Read accumulator
            acc = await self._issue_cmd(ctx, dut, funct3=1, funct7=0, in0=1, in1=0)
            assert acc == 100, f"expected acc=100, got {acc}"

            # SRDHM: acc * multiplier
            srdhm_result = await self._issue_cmd(
                ctx, dut,
                funct3=3, funct7=0,
                in0=acc & 0xFFFFFFFF, in1=multiplier & 0xFFFFFFFF,
            )
            expected_srdhm = ref_srdhm(acc, multiplier)
            assert srdhm_result == expected_srdhm, (
                f"srdhm: expected {expected_srdhm}, got {srdhm_result}"
            )

            # RDBPOT: shift right
            rdbpot_result = await self._issue_cmd(
                ctx, dut,
                funct3=4, funct7=0,
                in0=srdhm_result & 0xFFFFFFFF, in1=shift,
            )
            expected_rdbpot = ref_rdbpot(expected_srdhm, shift)
            assert rdbpot_result == expected_rdbpot, (
                f"rdbpot: expected {expected_rdbpot}, got {rdbpot_result}"
            )

        dut = Top()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("test_cfu_requant.vcd"):
            sim.run()

    def test_fallback_instruction(self):
        """Unused funct3 slots return in0 (fallback behavior)."""

        async def testbench(ctx):
            result = await self._issue_cmd(
                ctx, dut,
                funct3=7, funct7=0,  # slot 7 is unused
                in0=0xDEADBEEF, in1=0,
            )
            expected = to_signed32(0xDEADBEEF)
            assert result == expected, f"expected {expected}, got {result}"

        dut = Top()
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("test_cfu_fallback.vcd"):
            sim.run()
