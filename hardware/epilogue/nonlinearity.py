from amaranth import Module, Mux, Signal, signed
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out


class ReLU(wiring.Component):
    """
    ReLU activation: out = max(0, x).

    Combinational. Signed 32-bit input, signed 32-bit output.
    """

    def __init__(self, width=32):
        self.width = width
        super().__init__(
            {
                "x": In(signed(width)),
                "out": Out(signed(width)),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        m.d.comb += self.out.eq(Mux(self.x > 0, self.x, 0))
        return m


class ReLU6(wiring.Component):
    """
    ReLU6 activation: out = min(6, max(0, x)).

    Combinational. The upper bound of 6 is in quantized space —
    the actual clamp value depends on the quantization scale.
    For int8 quantized models, the clamp is typically at the
    requantized int8 level, not raw int32.

    clamp parameter sets the upper bound (default 6 for standard ReLU6,
    but the caller can set it to the quantized equivalent).
    """

    def __init__(self, width=32, clamp=6):
        self.width = width
        self.clamp = clamp
        super().__init__(
            {
                "x": In(signed(width)),
                "out": Out(signed(width)),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        clamped = Signal(signed(self.width))
        m.d.comb += clamped.eq(Mux(self.x > self.clamp, self.clamp, self.x))
        m.d.comb += self.out.eq(Mux(clamped > 0, clamped, 0))
        return m
