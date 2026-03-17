# Building a CFU (Custom Function Unit) ML Accelerator with Amaranth HDL

A practical guide for implementing [CFU-Playground](https://cfu-playground.readthedocs.io/en/latest/index.html)-style
accelerators using **Amaranth HDL** on the **Tang Nano 20K**.

---

## 1. What is a CFU?

A **Custom Function Unit** is accelerator hardware tightly coupled into a
RISC-V CPU pipeline. The CPU executes a custom opcode that:

1. Sends two 32-bit register values + a 10-bit function ID to the CFU
2. Waits for the CFU to respond
3. Writes the 32-bit result back into a destination register

```
         ┌──────────┐       cmd_valid / cmd_ready        ┌──────────┐
         │          │ ──── cmd_function_id[9:0] ────────► │          │
         │   CPU    │ ──── cmd_inputs_0[31:0]   ────────► │   CFU    │
         │ (VexRiscV│ ──── cmd_inputs_1[31:0]   ────────► │          │
         │  etc.)   │                                     │          │
         │          │ ◄──── rsp_valid / rsp_ready ─────── │          │
         │          │ ◄──── rsp_outputs_0[31:0]  ─────── │          │
         └──────────┘                                     └──────────┘
```

### CFU Bus Protocol

| Signal              | Direction  | Description                            |
|---------------------|-----------|----------------------------------------|
| `cmd_valid`         | CPU → CFU | CPU has a valid command                 |
| `cmd_ready`         | CFU → CPU | CFU can accept a command               |
| `cmd_function_id`   | CPU → CFU | `{funct7, funct3}` from the instruction|
| `cmd_inputs_0`      | CPU → CFU | Contents of `rs1`                      |
| `cmd_inputs_1`      | CPU → CFU | Contents of `rs2`                      |
| `rsp_valid`         | CFU → CPU | CFU has a valid response               |
| `rsp_ready`         | CPU → CFU | CPU can accept a response              |
| `rsp_outputs_0`     | CFU → CPU | Result to write to `rd`                |

**Handshake**: A transfer occurs when `valid & ready` are both high on the
same clock edge.

---

## 2. Architecture Overview

The CFU-Playground system looks like this:

```
┌─────────────────────────────────────────────────────┐
│                   FPGA SoC (LiteX)                  │
│                                                     │
│  ┌──────────┐    CFU Bus    ┌──────────────────┐    │
│  │ VexRiscV │◄────────────►│  Your CFU         │    │
│  │   CPU    │               │  (Amaranth HDL)   │    │
│  └────┬─────┘               └──────────────────┘    │
│       │                                             │
│  ┌────┴─────┐  ┌──────┐  ┌───────┐  ┌──────────┐  │
│  │  SRAM    │  │ UART │  │ Timer │  │ DDR DRAM │  │
│  └──────────┘  └──────┘  └───────┘  └──────────┘  │
└─────────────────────────────────────────────────────┘
```

**Key point**: The CFU has **no direct memory access**. The CPU must move
data to/from the CFU via register values.

---

## 3. Development Flow (Amaranth HDL)

### 3.1. Why Amaranth over Verilog?

| Feature           | Amaranth HDL            | Verilog                |
|-------------------|-------------------------|------------------------|
| Language          | Python DSL              | HDL                    |
| Unit testing      | Python `unittest`/pytest| Separate testbench     |
| Composition       | Python classes/modules  | Manual instantiation   |
| Simulation        | Built-in simulator      | External tools         |
| Output            | Generates Verilog       | Direct                 |

### 3.2. Project Structure

```
accel/
├── rtl/
│   ├── amaranth/           # Standalone Amaranth source
│   │   ├── top.py          # Top-level module
│   │   └── cfu.py          # CFU implementation
│   └── tests/              # Unit tests for RTL blocks
│       └── test_cfu.py
├── soc/                    # LiteX SoC build and integration
├── fw/                     # RISC-V firmware/apps
├── platforms/
│   └── tangnano20k/
│       └── constraints/
│           └── tangnano20k.cst
├── build/                  # Generated artifacts
├── docs/                   # Documentation
├── surfur/                 # Waveform viewer
└── Makefile                # Build system
```

### 3.3. Workflow

```
  Write Amaranth    →   Unit Test    →   Generate     →   Synthesise   →   Program
  (Python)              (pytest)         Verilog          (Yosys/PnR)      (openFPGALoader)
                           ↑                                                    │
                           └────── Profile on hardware, find bottlenecks ◄──────┘
```

---

## 4. Step-by-Step: Building a SIMD MAC Accelerator

This follows the CFU-Playground tutorial approach, but entirely in Amaranth.

### Step 1: Understand What to Accelerate

For a quantized int8 ML model (e.g. person detection), the inner loop of
convolution is:

```c
for (int i = 0; i < input_depth; i += 4) {
    acc += (input_data[i+0] + 128) * filter_data[i+0];
    acc += (input_data[i+1] + 128) * filter_data[i+1];
    acc += (input_data[i+2] + 128) * filter_data[i+2];
    acc += (input_data[i+3] + 128) * filter_data[i+3];
}
```

We can replace this with a single custom instruction that does **4 parallel
multiply-accumulates per cycle**.

### Step 2: Define the CFU Instruction

```
             7 bits
        +--------------+
funct7 = | (bool) reset |       ← non-zero resets accumulator
        +--------------+

              int8         int8         int8         int8
        +------------+------------+------------+------------+
  in0 = | input[0]   | input[1]   | input[2]   | input[3]   |
        +------------+------------+------------+------------+

              int8         int8         int8         int8
        +------------+------------+------------+------------+
  in1 = | filter[0]  | filter[1]  | filter[2]  | filter[3]  |
        +------------+------------+------------+------------+

                              int32
        +----------------------------------------------------+
output = | acc + Σ (input[i] + 128) × filter[i]               |
        +----------------------------------------------------+
```

### Step 3: Implement in Amaranth

See `rtl/amaranth/cfu.py` for the full implementation. Key points:

```python
from amaranth import *

class SimdMac(CfuBase):
    def elab(self, m, cmd_fire):
        bus = self.bus
        acc = Signal(signed(32))

        funct7 = bus.cmd_function_id[3:10]

        # 4 parallel byte multiplies
        prods = [Signal(signed(16), name=f"prod_{i}") for i in range(4)]
        for i, prod in enumerate(prods):
            in_byte = bus.cmd_inputs_0.word_select(i, 8).as_signed()
            filt_byte = bus.cmd_inputs_1.word_select(i, 8).as_signed()
            m.d.comb += prod.eq((in_byte + self.input_offset) * filt_byte)

        prod_sum = Signal(signed(32))
        m.d.comb += prod_sum.eq(prods[0] + prods[1] + prods[2] + prods[3])

        with m.If(cmd_fire):
            with m.If(funct7 != 0):
                m.d.sync += acc.eq(0)
                m.d.comb += bus.rsp_outputs_0.eq(0)
            with m.Else():
                m.d.sync += acc.eq(acc + prod_sum)
                m.d.comb += bus.rsp_outputs_0.eq(acc + prod_sum)
            m.d.comb += bus.rsp_valid.eq(1)
```

### Step 4: Write Tests

See `rtl/tests/test_cfu.py`. Use Amaranth's built-in simulator:

```python
async def process(ctx):
    # Reset accumulator
    await self._cmd(ctx, dut, funct7=1, in0=0, in1=0)

    # (1+128)*1 = 129
    result = await self._cmd(ctx, dut, funct7=0,
                             in0=pack_vals(1, 0, 0, 0),
                             in1=pack_vals(1, 0, 0, 0))
    assert result == 129
```

Run tests: `make test`

### Step 5: Use in Software (C side)

On the CPU side, the custom instruction is invoked via inline assembly:

```c
#include "cfu.h"

// In the convolution inner loop:
int32_t acc = cfu_op0(/*funct7=*/1, 0, 0);  // reset
for (int i = 0; i < input_depth; i += 4) {
    uint32_t in_word  = *(uint32_t*)&input_data[i];
    uint32_t flt_word = *(uint32_t*)&filter_data[i];
    acc = cfu_op0(/*funct7=*/0, in_word, flt_word);
}
```

### Step 6: Synthesise & Program

```bash
source ~/Projects/oss-cad-suite/environment

make synth    # Yosys → nextpnr → bitstream
make prog     # Program via USB (volatile)
make flash    # Program to flash (persistent)
```

---

## 5. Performance Expectations

From the CFU-Playground reference results (Arty A7):

| Stage                      | Inner Loop Cycles | Total Inference |
|----------------------------|-------------------|-----------------|
| Baseline (unoptimized)     | 977               | 163M            |
| Software specialization    | 431               | 117M (-28%)     |
| + Loop unrolling           | 431               | 117M            |
| + SIMD MAC CFU             | 178               | 86M  (-47%)     |

Actual numbers will vary on the Tang Nano 20K (different clock, FPGA fabric).

---

## 6. Going Further

- **Multi-cycle instructions**: For operations needing >1 cycle, deassert
  `cmd_ready` and assert `rsp_valid` when done
- **Pipeline multiple instructions**: Use `funct3` bits to select between
  different operations (up to 8 instructions per `funct7` group)
- **Memory components**: Use Amaranth `Memory` for lookup tables, caches,
  and delay buffers inside the CFU
- **Full SoC integration**: Use [LiteX](https://github.com/enjoy-digital/litex)
  to build a complete SoC with VexRiscV + your CFU

---

## 7. References

- [CFU-Playground Docs](https://cfu-playground.readthedocs.io/en/latest/index.html)
- [Amaranth HDL](https://github.com/amaranth-lang/amaranth)
- [Amaranth Tutorial (Vivonomicon)](https://vivonomicon.com/2020/04/14/learning-fpga-design-with-amaranth/)
- [oss-cad-suite](https://github.com/YosysHQ/oss-cad-suite-build)
- [Tang Nano 20K Wiki](https://wiki.sipeed.com/hardware/en/tang/tang-nano-20k/tang-nano-20k.html)
