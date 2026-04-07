from amaranth import Module, Signal, signed
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class SkewBuffer(wiring.Component):
    """
    Triangular delay buffer for systolic array skew/de-skew.

    depth ports, where port i is delayed by:
      - i cycles           (reverse=False, input skew)
      - (depth-1-i) cycles (reverse=True, output de-skew)

    Total registers: depth * (depth - 1) / 2.
    Port 0 with 0 delay is combinational (no register).

    Reusable for WS (1 buffer for activations) and OS (2 buffers:
    activations + weights).
    """

    def __init__(self, depth: int, width: int, *, reverse: bool = False):
        self.depth = depth
        self.width = width
        self.reverse = reverse

        ports = {}
        for i in range(depth):
            ports[f"in_{i}"] = In(signed(width))
            ports[f"out_{i}"] = Out(signed(width))

        super().__init__(ports)

    def elaborate(self, _platform):
        m = Module()

        for i in range(self.depth):
            delay = (self.depth - 1 - i) if self.reverse else i
            inp = getattr(self, f"in_{i}")
            out = getattr(self, f"out_{i}")

            if delay == 0:
                m.d.comb += out.eq(inp)
            else:
                stages = [Signal(signed(self.width), name=f"skew_{i}_s{j}")
                          for j in range(delay)]
                m.d.sync += stages[0].eq(inp)
                for j in range(1, delay):
                    m.d.sync += stages[j].eq(stages[j - 1])
                m.d.comb += out.eq(stages[-1])

        return m
