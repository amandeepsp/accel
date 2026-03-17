# Unit 1: Custom Compute — ALUs and Intrinsics

> **Course:** [00-architecture](00-architecture.md) > **[01-compute](01-compute.md)**

---

## 1.1 Every Accelerator Is a Fancy MAC Unit

Every ML accelerator ever built has one thing in common: a specialized multiply-accumulate unit.

- **NVIDIA Tensor Cores:** 4x4x4 matrix multiply-accumulate per cycle (FP16/INT8)
- **Google TPU:** 256x256 systolic array of MAC units
- **Apple Neural Engine:** 16-core design, each with a matrix multiply unit
- **Qualcomm Hexagon DSP:** 4-wide SIMD MAC instructions
- **Your CFU:** 4-lane INT8 SIMD MAC

The reason is arithmetic. A single layer of a neural network computes:

```
  output[i] = bias[i] + SUM_j(weight[i][j] * input[j])
```

That inner sum is a dot product — a sequence of multiply-accumulate operations. A convolution is many dot products. An attention layer is dot products with softmax. The entire inference workload reduces to: **multiply two numbers, add to a running sum, repeat billions of times.**

General-purpose CPUs can do this, but they waste most of their transistor budget on branch prediction, out-of-order execution, and cache coherency — machinery that's useless for a dot product loop. ML accelerators strip all of that away and pack the die with MAC units.

> **MLSys Connection:** When NVIDIA introduced Tensor Cores in the Volta architecture (2017), the key insight wasn't new math — it was dedicating silicon to a single operation (matrix multiply-accumulate) instead of using general ALUs. A V100 Tensor Core does a 4x4x4 FMA in one cycle. Your CFU's `SimdMac4` does a 1x4x1 dot product in one cycle. Same idea, different scale.

---

## 1.2 NVIDIA's dp4a: The Instruction You're Building

Before Tensor Cores, NVIDIA's Pascal architecture introduced `dp4a` — **dot product of 4x INT8, accumulated into INT32**:

```
  dp4a(a, b, c):
    c += a.byte[0] * b.byte[0]
       + a.byte[1] * b.byte[1]
       + a.byte[2] * b.byte[2]
       + a.byte[3] * b.byte[3]
    return c
```

Two 32-bit registers, each packed with four INT8 values. Multiply lane-by-lane, sum, accumulate. One instruction, four multiplies.

This is *exactly* what your `SimdMac4` does, with one addition: an input offset of 128 to handle TFLite's INT8 quantization convention.

```
  Your SimdMac4(a, b):
    acc += (a.byte[0] + 128) * b.byte[0]
         + (a.byte[1] + 128) * b.byte[1]
         + (a.byte[2] + 128) * b.byte[2]
         + (a.byte[3] + 128) * b.byte[3]
    return acc
```

> **MLSys Connection:** `dp4a` was a pivotal moment in ML hardware. It showed that INT8 inference could be 4x faster than FP32 with negligible accuracy loss for many models, because you pack 4 values where 1 used to be. This drove the entire quantization movement — and your project uses quantized INT8 for the same reason: it's the only way to fit useful work into your tiny FPGA.

---

## 1.3 Why the Offset?

INT8 quantization maps floating-point values to the range [-128, 127]. The formula is:

```
  real_value = scale * (int8_value - zero_point)
```

TFLite uses `zero_point = -128`, so before multiplying, the firmware adds 128. This shifts from signed INT8 to an unsigned domain:

```
  int8 value:   -128  -127  ...   0   ...  126   127
  after +128:      0     1  ... 128   ...  254   255
```

The offset is hardcoded to 128 in `hardware/mac.py`. Real models have per-layer zero-points that vary.

**Exercise:** If a layer's zero-point is -135 instead of -128, what happens with the hardcoded offset? Look at `self.INPUT_OFFSET = 128` in `hardware/mac.py`. The signal `self.input_offset` could be made dynamic — how would you expose it as a configurable parameter?

<details><summary>Hint 1</summary>

You would need a separate custom instruction (or a modifier on the existing one) to write a new offset value before running the MAC. Think about what funct3 values are available.

</details>

<details><summary>Hint 2</summary>

One approach: add a "set offset" instruction at a new funct7 slot. It takes the offset in rs1 and stores it in a register that `SimdMac4` reads instead of the constant. The Cfu class already supports up to 8 instruction slots via funct7.

</details>

<details><summary>Solution</summary>See solutions/01-unit/configurable-offset.py</details>

---

## 1.4 The R-Type Custom Instruction Encoding

The MAC is invoked via a custom RISC-V instruction using the `CUSTOM_0` opcode space (`0x0B`). The encoding is standard R-type:

```
  31        25 24    20 19    15 14  12 11     7 6       0
  ┌──────────┬────────┬────────┬──────┬────────┬─────────┐
  │  funct7  │  rs2   │  rs1   │funct3│   rd   │ opcode  │
  │  7 bits  │ 5 bits │ 5 bits │3 bits│ 5 bits │ 7 bits  │
  └──────────┴────────┴────────┴──────┴────────┴─────────┘
```

The encoding convention in this project:

- **funct7** selects the **instruction** (up to 128 slots). `mac4 = 0`.
- **funct3** is the **modifier** for that instruction (up to 8 modes). For mac4: `0 = accumulate`, `1 = reset + compute`.

In Zig (`firmware/src/cfu.zig`), this is encoded via inline assembly:

```
  .insn r CUSTOM_0, funct3, funct7, rd, rs1, rs2
```

In the hardware (`hardware/cfu.py`), the CFU dispatches on `funct7` to select which `Instruction` module handles the operation, then passes `funct3` through to the module as a per-instruction modifier.

> **MLSys Connection:** This is the same pattern as GPU ISA extensions. NVIDIA doesn't expose `dp4a` as a function call — it's a machine instruction (`IDP.4A` in the SASS ISA). It goes through the instruction decoder, gets dispatched to the right functional unit, and returns a result. Your `.insn r CUSTOM_0` is the RISC-V equivalent: a new instruction baked into the ISA that the CPU's decoder routes to your custom hardware.

**Exercise:** Decode the MAC instruction by hand. Given: `funct7=0x00`, `funct3=0b000` (accumulate mode), `rs1=x11 (a1)`, `rs2=x12 (a2)`, `rd=x10 (a0)`. Write out all 32 bits.

<details><summary>Hint 1</summary>

The fields, left to right: funct7 (7 bits), rs2 (5 bits), rs1 (5 bits), funct3 (3 bits), rd (5 bits), opcode (7 bits). Register numbers: x10=01010, x11=01011, x12=01100.

</details>

<details><summary>Hint 2</summary>

funct7=0000000, rs2=01100, rs1=01011, funct3=000, rd=01010, opcode=0001011. Concatenate them.

</details>

<details><summary>Solution</summary>See solutions/01-unit/instruction-decode.md</details>

**Exercise:** The Cfu currently has `SimdMac4` at funct7=0. You have 7 more instruction slots. What operations would you put in them? Think about the full inference pipeline: after a MAC, you need requantization (multiply by scale, shift right), clamping, bias addition.

<details><summary>Hint 1</summary>

Look at TFLite's quantized convolution post-processing: after the dot product accumulation, you apply (1) bias add, (2) multiply by a per-channel scale (SRDHM — Saturating Rounding Doubling High Multiply), (3) rounding right-shift (RDBPOT — Rounding Divide By Power Of Two), (4) clamp to output range.

</details>

<details><summary>Hint 2</summary>

A natural mapping: funct7=0 for mac4, funct7=1 for SRDHM, funct7=2 for RDBPOT. Each is a separate `Instruction` subclass registered in `elab_instructions()`. This is exactly what Part 3 (Autonomous) builds.

</details>

<details><summary>Solution</summary>See solutions/01-unit/instruction-plan.md</details>

---

## 1.5 The CFU Bus Protocol: How Your Device Accepts Compute

The CPU and CFU communicate via a **valid/ready handshake** — the same backpressure protocol used in AXI buses, PCIe, and virtually every hardware interface:

```
  CPU -> CFU:  cmd_valid, cmd_function_id[9:0], cmd_inputs_0, cmd_inputs_1
  CFU -> CPU:  cmd_ready, rsp_valid, rsp_outputs_0
  CPU -> CFU:  rsp_ready
```

A transfer occurs when `valid & ready` are both high on the same clock edge. The CFU uses a 3-state FSM:

```
  ┌──────────┐   cmd_valid & done     ┌──────────────┐
  │          │   & rsp_ready          │              │
  │ WAIT_CMD │<──────────────────────│ WAIT_TRANSFER│
  │          │                        │ (CPU not     │
  │ cmd_ready│   cmd_valid & done    │  ready yet)  │
  │ = 1      │   & !rsp_ready        │ rsp_valid=1  │
  │          │───────────────────────>│              │
  │          │                        └──────┬───────┘
  │          │   cmd_valid & !done           │ rsp_ready
  │          │──────────┐                    │
  └──────────┘          v            ┌───────┘
                ┌──────────────┐      │
                │WAIT_INSTRUCT.│──────┘
                │ (multi-cycle │  done
                │  instruction)│
                └──────────────┘
```

- **WAIT_CMD:** Idle. CFU asserts `cmd_ready`. When the CPU sends a command (`cmd_valid`), the instruction executes.
- **WAIT_INSTRUCTION:** Multi-cycle instruction is still computing. CFU deasserts `cmd_ready` so the CPU can't send another command.
- **WAIT_TRANSFER:** Instruction is done, result is buffered in `stored_output`, but the CPU hasn't read it yet (`rsp_ready` is low).

> **MLSys Connection:** This valid/ready handshake is identical in principle to how GPU command queues work. The host pushes commands; the device pulls them when ready. Backpressure prevents overflow. The difference is scale: a GPU has thousands of commands in flight across multiple queues. Your CFU processes one command at a time. But the protocol — "I have work" / "I'm ready for work" / "here's the result" / "I've consumed the result" — is universal.

**Exercise:** Why does the FSM need the `WAIT_TRANSFER` state? What would happen if the CFU completed a computation but immediately started accepting a new command before the CPU read the result?

<details><summary>Hint 1</summary>

Look at the `stored_output` register in `hardware/cfu.py`. What is it protecting?

</details>

<details><summary>Hint 2</summary>

Without `WAIT_TRANSFER`, the next instruction's output would overwrite the previous result before the CPU reads it. The stored output register holds the result until the CPU acknowledges it with `rsp_ready`. This is classic producer-consumer backpressure.

</details>

<details><summary>Solution</summary>See solutions/01-unit/fsm-analysis.md</details>

---

## 1.6 The Accumulator Pattern

Notice that `SimdMac4` maintains a running accumulator:

```python
  # funct3[0] selects base: 0 = accumulate, 1 = reset
  with m.If(self.funct3[0]):
      m.d.comb += base.eq(0)
  with m.Else():
      m.d.comb += base.eq(self.accumulator)
```

With `funct3=0` (accumulate mode), each call adds to the running sum. With `funct3=1` (reset mode), the accumulator starts from zero.

This matters because a typical convolution inner loop looks like:

```
  acc = 0                           # funct3=1 on first call
  for chunk in range(0, depth, 4):
      acc = mac4(acc, input[chunk:chunk+4], weight[chunk:chunk+4])  # funct3=0
  output = requantize(acc)
```

The accumulator lets you process a dot product in chunks of 4, without moving intermediate results back to the CPU between chunks.

> **MLSys Connection:** This is the same reason GPU Tensor Cores accumulate into a register file rather than writing intermediate results to shared memory. Memory traffic is the bottleneck in ML workloads, not compute. The accumulator keeps the running sum in the fastest possible storage (a hardware register), avoiding a round-trip through memory on every iteration. NVIDIA calls this "register-level accumulation" and it's critical to Tensor Core throughput.

**Exercise:** The `cfu.zig` wrapper `mac4(acc, a, b)` returns `acc + cfu_call(...)` — it accumulates in *software*. But the hardware also has an internal accumulator. When would you use one vs. the other?

<details><summary>Hint 1</summary>

Consider: what happens if you need to compute two independent dot products back-to-back? The hardware accumulator doesn't know they're separate.

</details>

<details><summary>Hint 2</summary>

The hardware accumulator is ideal for a single long dot product — you don't pay the overhead of reading the result back to a CPU register between iterations. The software accumulator gives you explicit control: you can compute multiple independent sums, or inspect intermediate values. In Part 3 (Autonomous), the hardware accumulator becomes essential because the CPU isn't in the loop at all.

</details>

<details><summary>Solution</summary>See solutions/01-unit/accumulator-tradeoff.md</details>

---

## 1.7 Exercise: Build SimdMac4

The project already has the MAC4 implementation in `hardware/mac.py`. This exercise is retrospective — understand what was built by rebuilding it.

**Spec:** Build a `SimdMac4` module in Amaranth that:
1. Takes two 32-bit inputs, each packed with four INT8 values
2. Adds an offset of 128 to each byte of input 0
3. Multiplies corresponding byte lanes
4. Sums the four products
5. Accumulates into a running sum (controlled by `funct3[0]`: 0=accumulate, 1=reset)
6. Completes in a single cycle (`done` always high)

**Constraints:**
- Use `Signal.word_select(i, 8)` to extract byte lanes
- Products must be 32-bit to avoid overflow: (255 + 128) * 255 = 97,665, which exceeds 16 bits
- The accumulator is a sync register (updates on clock edge), but the output is combinational (available same cycle)

<details><summary>Hint 1</summary>

Start with the interface. Your module extends `Instruction`, which gives you: `self.in0`, `self.in1` (32-bit inputs), `self.funct3`, `self.output` (32-bit), `self.done`, `self.start`. You need to add `self.accumulator = Signal(32)`.

</details>

<details><summary>Hint 2</summary>

The core logic: create 4 product signals. For each lane i, extract byte i from in0 and in1, compute `(a + 128) * b`, and store in a product signal. Then `self.output.eq(base + p0 + p1 + p2 + p3)` where base depends on funct3[0].

</details>

<details><summary>Hint 3</summary>

Don't forget: the accumulator must update on `self.start` (not every cycle), and the output must be combinational so it's ready in the same cycle the command arrives. This is why `done` is always 1 — it's a single-cycle instruction.

</details>

<details><summary>Solution</summary>See hardware/mac.py — the existing implementation is the reference.</details>

---

## 1.8 GPU Parallel: Your MAC4 IS dp4a

To put it concretely:

| Property | NVIDIA dp4a | Your SimdMac4 |
|---|---|---|
| Inputs | 2x 32-bit (4x INT8 packed) | 2x 32-bit (4x INT8 packed) |
| Output | 1x 32-bit (INT32 accumulator) | 1x 32-bit (INT32 accumulator) |
| Operations per cycle | 4 MACs | 4 MACs |
| Input offset | None (handled in software) | 128 (hardcoded, for TFLite quantization) |
| Accumulator | Register-level | Hardware register, selectable reset |
| Invocation | PTX instruction `dp4a.atype.btype` | `.insn r CUSTOM_0, funct3, 0, rd, rs1, rs2` |
| Parallelism | 1000s of SMs, each with 64+ dp4a units | 1 CFU, 1 MAC unit |

A single A100 GPU has 6,912 CUDA cores, each capable of `dp4a`. You have one. But the architecture of that one unit — pack values, multiply lanes, sum, accumulate — is identical.

Tensor Cores go further: instead of a 1x4 dot product, they do a 4x4x4 matrix multiply in one cycle. That's 64 MACs per Tensor Core per cycle. But if you unroll it, each one is doing what your `SimdMac4` does, just spatially replicated and interconnected.

> **MLSys Connection:** The progression from `dp4a` (1x4 dot product) to Tensor Cores (4x4x4 matmul) to the H100's FP8 Tensor Cores (bigger matrices, lower precision) is the same progression this course follows: Tier 2 (your SIMD MAC) -> Tier 4 (systolic array in Unit 5). You're walking the same design path NVIDIA did from Pascal to Hopper, at FPGA scale.

---

## 1.9 Checkpoint

Before moving to Unit 2, verify:

- [ ] I understand why ML accelerators are fundamentally MAC units (the inner loop of inference is dot products)
- [ ] I can explain what `dp4a` does and how `SimdMac4` is the same operation with an offset
- [ ] I understand the R-type encoding: funct7 selects the instruction (mac4=0), funct3 is the modifier (0=accumulate, 1=reset)
- [ ] I can trace the CFU bus FSM: WAIT_CMD -> instruction executes -> result returned via valid/ready handshake
- [ ] I know why the accumulator pattern exists (reduce memory traffic in dot product loops)
- [ ] The MAC hardware passes simulation tests (`python -m pytest hardware/`)
- [ ] I've read `hardware/mac.py` and `hardware/cfu.py` and understand every line

---

## Suggested Readings

- **tinygrad source:** `tinygrad/runtime/ops_cuda.py` — look at how tinygrad generates CUDA kernels. The inner loops compile down to instructions like `dp4a`.
- **tinygrad source:** `tinygrad/codegen/kernel.py` — the kernel code generator. Notice how it decides vector widths and accumulator types.
- **Blog:** Lei Mao, ["NVIDIA dp4a"](https://leimao.github.io/blog/NVIDIA-dp4a/) — detailed breakdown of the dp4a instruction, including PTX assembly and performance numbers.
- **Paper:** Jouppi et al., ["In-Datacenter Performance Analysis of a Tensor Processing Unit"](https://arxiv.org/abs/1704.04760) (2017) — Section 3 describes the TPU's matrix multiply unit (a 256x256 systolic array of MACs). Your `SimdMac4` is one cell of this array.
- **Blog:** Bruce Hoult, ["Custom RISC-V Instructions"](https://hoult.org/riscv_custom_instructions.html) — practical guide to the CUSTOM_0 opcode space and inline assembly encoding.
- **Reference:** CFU-Playground, [`proj/avg_pdti8/`](https://github.com/google/CFU-Playground/tree/main/proj/avg_pdti8) — a SIMD MAC very similar to yours, with TFLite integration. Compare their instruction encoding to yours.
- **Paper:** Markidis et al., ["NVIDIA Tensor Core Programmability, Performance & Precision"](https://arxiv.org/abs/1803.04014) (2018) — how Tensor Cores evolved from dp4a-style operations to matrix-level instructions.

---

**Previous:** [Unit 0: Architecture](00-architecture.md)
