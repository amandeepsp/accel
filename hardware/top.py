"""CFU top-level with CPU-sequencer local stores."""

from amaranth import ClockSignal, ResetSignal, Signal
from amaranth.back.verilog import convert
from amaranth.lib.memory import Memory
from cfu import Cfu
from epilogue.quant import RoundingDividebyPOTInstruction, SRDHMInstruction


class Top(Cfu):
    STORE_DEPTH = 512

    def elab_instructions(self, m):
        m.submodules["srdhm"] = srdhm = SRDHMInstruction()
        m.submodules["rdpot"] = rdpot = RoundingDividebyPOTInstruction()

        return {3: srdhm, 4: rdpot}


if __name__ == "__main__":
    import re

    top = Top()
    ports = top.ports + [ClockSignal("sync"), ResetSignal("sync")]
    v = convert(top, name="Cfu", ports=ports, strip_internal_attrs=True)
    # Yosys emits dump_module debug regs that break GTKWave's VCD parser.
    v = re.sub(r"^.*dump_module.*\n", "", v, flags=re.MULTILINE)
    with open("top.v", "w") as f:
        f.write(v)
    print("Wrote top.v")
