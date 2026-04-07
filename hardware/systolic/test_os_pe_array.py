from amaranth.sim import Simulator

from os_pe_array import OutputStationaryPEArray


class TestOutputStationaryPEArray:
    def test_2x2_matmul(self):
        """
        C = A @ B, 2x2 OS array. C-wide drain output.
          A = [[1, 2],   B = [[5, 6],
               [3, 4]]        [7, 8]]
          C = [[19, 22],
               [43, 50]]

        Drain reads bottom row first (row 1), then row 0 shifts down.
        """

        async def testbench(ctx):
            ctx.set(dut.psum_load, 1)
            await ctx.tick()
            ctx.set(dut.psum_load, 0)

            # Feed K=2 elements (all rows/cols simultaneously, skew handles timing)
            ctx.set(dut.act_in_0, 1)
            ctx.set(dut.act_in_1, 3)
            ctx.set(dut.w_in_0, 5)
            ctx.set(dut.w_in_1, 6)
            await ctx.tick()

            ctx.set(dut.act_in_0, 2)
            ctx.set(dut.act_in_1, 4)
            ctx.set(dut.w_in_0, 7)
            ctx.set(dut.w_in_1, 8)
            await ctx.tick()

            # Flush: R + C - 2 = 2 cycles
            ctx.set(dut.act_in_0, 0)
            ctx.set(dut.act_in_1, 0)
            ctx.set(dut.w_in_0, 0)
            ctx.set(dut.w_in_1, 0)
            await ctx.tick()
            await ctx.tick()

            # Drain: row 1 (bottom) exits first, then row 0
            ctx.set(dut.drain, 1)
            await ctx.delay(1e-7)
            # Before first drain tick: bottom row has its own values
            assert ctx.get(dut.psum_out_0) == 43  # C[1][0]
            assert ctx.get(dut.psum_out_1) == 50  # C[1][1]

            await ctx.tick()
            # Row 0 shifted down into bottom row
            assert ctx.get(dut.psum_out_0) == 19  # C[0][0]
            assert ctx.get(dut.psum_out_1) == 22  # C[0][1]
            ctx.set(dut.drain, 0)

        dut = OutputStationaryPEArray(2, 2)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_os_array_2x2.vcd"):
            sim.run()

    def test_2x2_with_negatives(self):
        """
        Signed matmul with C-wide drain.
          A = [[ 1, -2],   B = [[-3,  4],
               [ 5,  6]]        [ 7, -8]]
          C = [[-17, 20],
               [27, -28]]
        """

        async def testbench(ctx):
            A = [[1, -2], [5, 6]]
            B = [[-3, 4], [7, -8]]

            ctx.set(dut.psum_load, 1)
            await ctx.tick()
            ctx.set(dut.psum_load, 0)

            for k in range(2):
                ctx.set(dut.act_in_0, A[0][k])
                ctx.set(dut.act_in_1, A[1][k])
                ctx.set(dut.w_in_0, B[k][0])
                ctx.set(dut.w_in_1, B[k][1])
                await ctx.tick()

            ctx.set(dut.act_in_0, 0)
            ctx.set(dut.act_in_1, 0)
            ctx.set(dut.w_in_0, 0)
            ctx.set(dut.w_in_1, 0)
            await ctx.tick()
            await ctx.tick()

            # Drain: bottom row first
            ctx.set(dut.drain, 1)
            await ctx.delay(1e-7)
            assert ctx.get(dut.psum_out_0) == 27   # C[1][0]
            assert ctx.get(dut.psum_out_1) == -28   # C[1][1]

            await ctx.tick()
            assert ctx.get(dut.psum_out_0) == -17   # C[0][0]
            assert ctx.get(dut.psum_out_1) == 20    # C[0][1]
            ctx.set(dut.drain, 0)

        dut = OutputStationaryPEArray(2, 2)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_os_array_neg.vcd"):
            sim.run()

    def test_4x4_matmul(self):
        """
        4x4 OS array, K=4, C-wide drain (4 cycles to read all rows).
          A = [[1, 0, 2, -1],    B = [[ 1, 2, 0, -1],
               [0, 3, -1, 0],         [ 0, 1, 3,  2],
               [2, 1, 0, 4],          [-1, 0, 1,  0],
               [-1, 0, 1, 2]]         [ 2, -1, 0,  3]]
          C = [[-3, 3, 2, -4],
               [ 1, 3, 8,  6],
               [10, 1, 3, 12],
               [ 2, -4, 1, 7]]
        """

        async def testbench(ctx):
            A = [[ 1, 0, 2,-1],
                 [ 0, 3,-1, 0],
                 [ 2, 1, 0, 4],
                 [-1, 0, 1, 2]]
            B = [[ 1, 2, 0,-1],
                 [ 0, 1, 3, 2],
                 [-1, 0, 1, 0],
                 [ 2,-1, 0, 3]]
            expected = [[-3, 3, 2,-4],
                        [ 1, 3, 8, 6],
                        [10, 1, 3,12],
                        [ 2,-4, 1, 7]]

            ctx.set(dut.psum_load, 1)
            await ctx.tick()
            ctx.set(dut.psum_load, 0)

            for k in range(4):
                for r in range(4):
                    ctx.set(getattr(dut, f"act_in_{r}"), A[r][k])
                for c in range(4):
                    ctx.set(getattr(dut, f"w_in_{c}"), B[k][c])
                await ctx.tick()

            for r in range(4):
                ctx.set(getattr(dut, f"act_in_{r}"), 0)
            for c in range(4):
                ctx.set(getattr(dut, f"w_in_{c}"), 0)
            for _ in range(6):
                await ctx.tick()

            # Drain: bottom row (3) first, then 2, 1, 0
            ctx.set(dut.drain, 1)
            for drain_cycle in range(4):
                row = 3 - drain_cycle
                if drain_cycle == 0:
                    await ctx.delay(1e-7)
                else:
                    await ctx.tick()
                for c in range(4):
                    val = ctx.get(getattr(dut, f"psum_out_{c}"))
                    assert val == expected[row][c], \
                        f"C[{row}][{c}] = {val}, expected {expected[row][c]}"
            ctx.set(dut.drain, 0)

        dut = OutputStationaryPEArray(4, 4)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_os_array_4x4.vcd"):
            sim.run()

    def test_psum_load_between_tiles(self):
        """Reset accumulators between tiles and compute again."""

        async def testbench(ctx):
            # First tile: A=[[1,2],[3,4]], B=I → C=A
            ctx.set(dut.psum_load, 1)
            await ctx.tick()
            ctx.set(dut.psum_load, 0)

            ctx.set(dut.act_in_0, 1)
            ctx.set(dut.act_in_1, 3)
            ctx.set(dut.w_in_0, 1)
            ctx.set(dut.w_in_1, 0)
            await ctx.tick()
            ctx.set(dut.act_in_0, 2)
            ctx.set(dut.act_in_1, 4)
            ctx.set(dut.w_in_0, 0)
            ctx.set(dut.w_in_1, 1)
            await ctx.tick()
            ctx.set(dut.act_in_0, 0)
            ctx.set(dut.act_in_1, 0)
            ctx.set(dut.w_in_0, 0)
            ctx.set(dut.w_in_1, 0)
            await ctx.tick()
            await ctx.tick()

            # Drain first tile
            ctx.set(dut.drain, 1)
            await ctx.delay(1e-7)
            assert ctx.get(dut.psum_out_0) == 3   # C[1][0]
            assert ctx.get(dut.psum_out_1) == 4   # C[1][1]
            await ctx.tick()
            assert ctx.get(dut.psum_out_0) == 1   # C[0][0]
            assert ctx.get(dut.psum_out_1) == 2   # C[0][1]
            ctx.set(dut.drain, 0)

            # Reset and second tile: same A, B=[[2,0],[0,3]]
            ctx.set(dut.psum_load, 1)
            await ctx.tick()
            ctx.set(dut.psum_load, 0)

            ctx.set(dut.act_in_0, 1)
            ctx.set(dut.act_in_1, 3)
            ctx.set(dut.w_in_0, 2)
            ctx.set(dut.w_in_1, 0)
            await ctx.tick()
            ctx.set(dut.act_in_0, 2)
            ctx.set(dut.act_in_1, 4)
            ctx.set(dut.w_in_0, 0)
            ctx.set(dut.w_in_1, 3)
            await ctx.tick()
            ctx.set(dut.act_in_0, 0)
            ctx.set(dut.act_in_1, 0)
            ctx.set(dut.w_in_0, 0)
            ctx.set(dut.w_in_1, 0)
            await ctx.tick()
            await ctx.tick()

            ctx.set(dut.drain, 1)
            await ctx.delay(1e-7)
            assert ctx.get(dut.psum_out_0) == 6    # C[1][0]
            assert ctx.get(dut.psum_out_1) == 12   # C[1][1]
            await ctx.tick()
            assert ctx.get(dut.psum_out_0) == 2    # C[0][0]
            assert ctx.get(dut.psum_out_1) == 6    # C[0][1]
            ctx.set(dut.drain, 0)

        dut = OutputStationaryPEArray(2, 2)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_os_array_tiles.vcd"):
            sim.run()

    def test_2x2_k_tiled_gemm(self):
        """
        K-tiled GEMM on OS array. Same data as WS K-tiled test.
          A = [[1, 2, 3, 4],    W = [[ 1, -1],
               [5, 6, 7, 8]]         [ 2,  0],
                                      [-1,  3],
                                      [ 0,  2]]
          A @ W = [[ 2, 16],
                   [10, 32]]

        2x2 OS array, K=2 per tile. Two K-tiles per pass, externally
        accumulated. Unlike WS, OS processes all M-rows in one pass
        but needs K-tiling when K > array size.

        K-tile 0: A[:,0:2] @ W[0:2,:]   K-tile 1: A[:,2:4] @ W[2:4,:]
        External accum sums the two drain results per cell.
        """

        async def testbench(ctx):
            A = [[1, 2, 3, 4],
                 [5, 6, 7, 8]]
            W = [[ 1, -1],
                 [ 2,  0],
                 [-1,  3],
                 [ 0,  2]]
            expected = [[2, 16], [10, 32]]

            accum = [[0, 0], [0, 0]]

            # === K-tile 0: A[:,0:2] @ W[0:2,:] ===
            ctx.set(dut.psum_load, 1)
            await ctx.tick()
            ctx.set(dut.psum_load, 0)

            for k in range(2):
                ctx.set(dut.act_in_0, A[0][k])
                ctx.set(dut.act_in_1, A[1][k])
                ctx.set(dut.w_in_0, W[k][0])
                ctx.set(dut.w_in_1, W[k][1])
                await ctx.tick()

            ctx.set(dut.act_in_0, 0)
            ctx.set(dut.act_in_1, 0)
            ctx.set(dut.w_in_0, 0)
            ctx.set(dut.w_in_1, 0)
            await ctx.tick()
            await ctx.tick()

            # Drain K-tile 0: bottom row (M=1) first, then top (M=0)
            ctx.set(dut.drain, 1)
            await ctx.delay(1e-7)
            accum[1][0] += ctx.get(dut.psum_out_0)
            accum[1][1] += ctx.get(dut.psum_out_1)
            await ctx.tick()
            accum[0][0] += ctx.get(dut.psum_out_0)
            accum[0][1] += ctx.get(dut.psum_out_1)
            ctx.set(dut.drain, 0)

            # === K-tile 1: A[:,2:4] @ W[2:4,:] ===
            ctx.set(dut.psum_load, 1)
            await ctx.tick()
            ctx.set(dut.psum_load, 0)

            for k in range(2):
                ctx.set(dut.act_in_0, A[0][k + 2])
                ctx.set(dut.act_in_1, A[1][k + 2])
                ctx.set(dut.w_in_0, W[k + 2][0])
                ctx.set(dut.w_in_1, W[k + 2][1])
                await ctx.tick()

            ctx.set(dut.act_in_0, 0)
            ctx.set(dut.act_in_1, 0)
            ctx.set(dut.w_in_0, 0)
            ctx.set(dut.w_in_1, 0)
            await ctx.tick()
            await ctx.tick()

            # Drain K-tile 1
            ctx.set(dut.drain, 1)
            await ctx.delay(1e-7)
            accum[1][0] += ctx.get(dut.psum_out_0)
            accum[1][1] += ctx.get(dut.psum_out_1)
            await ctx.tick()
            accum[0][0] += ctx.get(dut.psum_out_0)
            accum[0][1] += ctx.get(dut.psum_out_1)
            ctx.set(dut.drain, 0)

            for r in range(2):
                for c in range(2):
                    assert accum[r][c] == expected[r][c], \
                        f"C[{r}][{c}] = {accum[r][c]}, expected {expected[r][c]}"

        dut = OutputStationaryPEArray(2, 2)
        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        with sim.write_vcd("waves/test_os_array_k_tiled.vcd"):
            sim.run()
