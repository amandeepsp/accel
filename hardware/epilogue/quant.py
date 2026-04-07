from amaranth import Elaboratable, Module, Mux, Signal, signed

from cfu import Instruction

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


class SRDHM(Elaboratable):
    def __init__(self) -> None:
        self.a = Signal(signed(32))
        self.b = Signal(signed(32))
        self.start = Signal()
        self.out = Signal(signed(32))
        self.done = Signal()

    def elaborate(self, platform):
        m = Module()

        m.d.sync += self.done.eq(0)
        ab = Signal(signed(64))
        nudge = 1 << 30
        with m.FSM():
            with m.State("stage0"):
                with m.If(self.start):
                    with m.If((self.a == INT32_MIN) & (self.b == INT32_MIN)):
                        m.d.sync += [
                            self.out.eq(INT32_MAX),
                            self.done.eq(1),
                        ]
                    with m.Else():
                        m.d.sync += ab.eq(self.a * self.b)
                        m.next = "stage1"
            with m.State("stage1"):
                m.d.sync += [
                    self.out.eq((ab + nudge)[31:]),
                    self.done.eq(1),
                ]
                m.next = "stage0"

        return m


class SRDHMInstruction(Instruction):
    def elaborate(self, platform):
        m = super().elaborate(platform)
        m.submodules["srdhm"] = srdhm = SRDHM()
        m.d.comb += [
            srdhm.a.eq(self.in0s),
            srdhm.b.eq(self.in1s),
            srdhm.start.eq(self.start),
            self.output.eq(srdhm.out),
            self.done.eq(srdhm.done),
        ]
        return m


class RoundingDividebyPOT(Elaboratable):
    """
    This divides by a power of two, rounding to the nearest whole number.
    """

    def __init__(self):
        self.x = Signal(signed(32))
        self.exponent = Signal(5)
        self.result = Signal(signed(32))

    def elaborate(self, platform):
        m = Module()
        mask = (1 << self.exponent) - 1
        remainder = self.x & mask
        threshold = (mask >> 1) + self.x[31]
        rounding = Mux(remainder > threshold, 1, 0)
        m.d.comb += self.result.eq((self.x >> self.exponent) + rounding)
        return m


class RoundingDividebyPOTInstruction(Instruction):
    def elaborate(self, platform):
        m = super().elaborate(platform)
        rdbypot = RoundingDividebyPOT()
        m.submodules["RDByPOT"] = rdbypot
        m.d.comb += [
            rdbypot.x.eq(self.in0s),
            rdbypot.exponent.eq(self.in1),
            self.output.eq(rdbypot.result),
            self.done.eq(1),
        ]
        return m
