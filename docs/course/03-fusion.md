# Unit 3: Kernel Fusion — Why Round Trips Kill You

> **Course:** [01-compute](01-compute.md) | [02-datapath](02-datapath.md) | **03-fusion** | [04-autonomy](04-autonomy.md)
>
> Learn how ML compilers and GPU hardware work, by building a tiny version on a Tang Nano 20K FPGA.

---

## The Real-World Problem

In a naive ML framework, the expression `relu(batchnorm(conv(x)))` becomes three separate kernel launches:

```
  Kernel 1: conv(x)      → write result to DRAM
  Kernel 2: batchnorm(y) → read from DRAM, write result to DRAM
  Kernel 3: relu(z)      → read from DRAM, write result to DRAM
```

Each kernel reads its input from memory and writes its output back. For a 1024-element tensor in FP32, that is 4 KB read + 4 KB written *per kernel* — 24 KB of memory traffic for what could be done in a single pass reading 4 KB and writing 4 KB (8 KB total).

The compute is cheap. The memory traffic is not. On an A100 GPU:

- Compute throughput: 312 TFLOPS (FP16)
- Memory bandwidth: 2 TB/s
- Arithmetic intensity threshold: ~156 FLOPs/byte

If your operation does fewer than 156 FLOPs per byte of data it touches, you are **memory-bound**. Element-wise ops like ReLU, addition, and requantisation are all massively memory-bound. Fusing them with the preceding compute kernel eliminates the intermediate memory round trips entirely.

This is why XLA, `torch.compile`, TVM, and Triton all have fusion passes as their most important optimisation. Fusion doesn't make compute faster — it eliminates memory traffic.

> **🔗 MLSys Connection:** XLA's fusion pass identifies "fusible" operations — chains of element-wise ops following a reduce (like matmul). It generates a single HLO fusion kernel that computes the entire chain without writing intermediates to global memory. `torch.compile` via Inductor does the same thing, generating Triton kernels that fuse pointwise ops into reduction epilogues. The pattern is universal: keep data in registers/shared memory, avoid global memory round trips.

---

## Fusion in Your System

On your FPGA accelerator, what plays the role of "DRAM"?

**The UART link.**

Every time an intermediate result leaves the CFU, travels back over UART to the host, gets processed in Python, and comes back — that is the equivalent of a GPU kernel writing to DRAM and the next kernel reading it back.

What plays the role of "registers" or "shared memory"?

**The CFU's internal accumulator and pipeline registers.**

If you can keep data inside the CFU — computing MAC, then requantising, then clamping — without ever sending intermediate results back to the host, you have *fused* those operations. One round trip instead of three.

```
  UNFUSED (3 round trips per output element):

  Host ──MAC4──► CFU ──result──► Host    (round trip 1: ~2.4 ms)
  Host ──SRDHM─► CFU ──result──► Host    (round trip 2: ~2.4 ms)
  Host ──RDBPOT► CFU ──result──► Host    (round trip 3: ~2.4 ms)
  Host clamps result in Python             Total: ~7.2 ms

  FUSED (1 round trip per output element):

  Host ──MAC4──► CFU
                 CFU: MAC4 → SRDHM → RDBPOT → clamp
                 CFU ──INT8 result──► Host    (round trip 1: ~2.4 ms)
                                               Total: ~2.4 ms
```

Fusion gives you a 3x latency reduction for free — no faster hardware, no wider datapath. Just keeping data on-chip.

> **🔗 MLSys Connection:** This is exactly what happens when `torch.compile` fuses `linear + relu` into a single Triton kernel. The matmul result stays in GPU registers, ReLU is applied in the epilogue, and only the final result is written to global memory. The 3x saving from eliminating two memory round trips is typical; for longer chains (conv + batchnorm + relu + add), the savings are even larger.

---

## The TFLite INT8 Quantisation Pipeline

To understand *what* to fuse, you need to understand the math. INT8 quantised inference in TFLite works as follows:

### The Convolution Output Transform

After the MAC loop accumulates `input_depth` products into an INT32 accumulator, the result must be converted back to INT8 for the next layer:

```
  acc (INT32, from MAC loop)
    │
    ├── + bias               per-channel INT32 bias
    │
    ├── SRDHM(acc, mult)     SaturatingRoundingDoubleHighMul
    │                        Rescale by a per-channel multiplier
    │                        Effectively: acc × M where M ∈ [0.5, 1.0)
    │
    ├── RDBPOT(acc, shift)   RoundingDivideByPowerOfTwo
    │                        Right-shift by a per-channel amount
    │
    ├── + output_offset      per-layer constant (typically -128 or 0)
    │
    └── clamp(-128, 127)     Saturate to INT8 range
```

Without hardware fusion, each of these steps is a separate host round trip. With fusion, the entire pipeline executes inside the CFU in a handful of clock cycles.

### Why Software Requantisation is Expensive

On a RV32IM core (no 64-bit multiply), SRDHM alone costs ~10-15 cycles per element:

```
  // Software SRDHM on RV32IM:
  int64_t product = (int64_t)a * (int64_t)b;    // 4-8 cycles (no HW mul64!)
  int32_t high = (int32_t)(product >> 31);       // multi-cycle shift
  // + nudge, saturation check...

  // Software RDBPOT:
  // + shift, rounding, sign handling...

  // Total: ~10-15 CPU cycles per output element
  // For a layer producing 2304 outputs: ~30,000 wasted CPU cycles
```

With hardware fusion: 0 CPU cycles. The CFU's DSP blocks do the 64-bit multiply in 1 clock cycle.

> **🔗 MLSys Connection:** This is directly analogous to how GPU tensor cores fuse the accumulation (FP16 multiply, FP32 accumulate) with the output conversion (FP32 to FP16/BF16/INT8). The "epilogue" of a GEMM kernel — bias add, activation function, requantisation — is fused into the same kernel that does the matrix multiply. Writing a separate kernel for each step would be correct but catastrophically slow. Google's `hps_accel` CFU implements this as a 7-stage PostProcess pipeline: SRDHM, RDBPOT, clamp, pack — all in hardware, fully pipelined at 1 output per cycle.

---

## Exercise 1: Understand SRDHM Mathematically

Before writing any hardware, implement SRDHM in Python. You must understand the math before you can verify hardware correctness.

**Spec: SaturatingRoundingDoubleHighMul(a, b)**

```
  1. Compute the 64-bit product:  product = int64(a) * int64(b)
  2. Add the rounding nudge:      product = product + (1 << 30)
  3. Extract upper 32 bits:       result  = int32(product >> 31)
  4. Saturation: if a == INT32_MIN and b == INT32_MIN → return INT32_MAX
```

The name explains the operation:
- **Saturating**: handles the one overflow case (INT32_MIN * INT32_MIN)
- **Rounding**: the `+ (1 << 30)` nudge rounds to nearest instead of truncating
- **Double High Mul**: takes the upper 32 bits of a 64-bit product, but shifted by 31 (not 32) — effectively multiplying by 2 (the "double")

**Exercise 1.1:** Implement `srdhm(a: int, b: int) -> int` in Python.

<details><summary>Hint 1</summary>

Python has arbitrary-precision integers, so the 64-bit multiply is trivial. The tricky part is getting the sign handling and saturation right. Use `numpy.int32` or manual masking to simulate 32-bit overflow.

</details>

<details><summary>Hint 2</summary>

The saturation case: `INT32_MIN * INT32_MIN` produces a positive number that overflows INT32. Detect this *before* the multiply: if both inputs equal `-(1<<31)`, return `(1<<31) - 1` immediately.

</details>

**Test vectors for SRDHM:**

| a | b | Expected | Why |
|---|---|---|---|
| `1` | `1` | `0` | Product = 1 + nudge = 1073741825. >> 31 = 0 (too small) |
| `0x40000000` (1073741824) | `2` | `1` | Product = 2147483648 + nudge = 3221225472. >> 31 = 1 |
| `INT32_MIN` (-2147483648) | `INT32_MIN` (-2147483648) | `INT32_MAX` (2147483647) | Saturation case |
| `0x40000000` | `0x40000000` | `0x20000000` (536870912) | Product = 2^60 + nudge. >> 31 = 2^29 |
| `-1` | `1` | `0` | Product = -1 + nudge = 1073741823. >> 31 = 0 |

<details><summary>Solution</summary>See `solutions/03-fusion/srdhm.py` for a reference implementation.</details>

---

## Exercise 2: Understand RDBPOT Mathematically

**Spec: RoundingDivideByPowerOfTwo(a, shift)**

This is an arithmetic right shift with rounding toward zero (not toward negative infinity, which is what a plain `>>` gives for negative numbers).

```
  mask      = (1 << shift) - 1
  remainder = a & mask
  threshold = (mask >> 1) + (1 if a < 0 else 0)
  result    = (a >> shift) + (1 if remainder > threshold else 0)
```

The intuition: for positive numbers, this rounds to nearest (round half up). For negative numbers, the `+1` in the threshold biases rounding toward zero, matching the behaviour specified by TFLite's quantisation scheme.

**Exercise 2.1:** Implement `rdbpot(a: int, shift: int) -> int` in Python.

<details><summary>Hint 1</summary>

The `>> shift` for negative numbers in Python gives floor division (rounds toward negative infinity). The correction term `(1 if remainder > threshold else 0)` bumps the result by 1 when the remainder is large enough, effectively rounding toward zero instead.

</details>

<details><summary>Hint 2</summary>

Be careful with Python's arbitrary-precision integers. You need to simulate 32-bit signed arithmetic. Use `a & 0xFFFFFFFF` and sign-extend when needed, or use `numpy.int32`.

</details>

**Test vectors for RDBPOT:**

| a | shift | Expected | Why |
|---|---|---|---|
| `16` | `3` | `2` | 16 >> 3 = 2. Remainder = 0, no rounding needed. |
| `17` | `3` | `2` | 17 >> 3 = 2. Remainder = 1, threshold = 3. 1 <= 3, no round up. |
| `20` | `3` | `3` | 20 >> 3 = 2. Remainder = 4, threshold = 3. 4 > 3, round up to 3. |
| `-17` | `3` | `-2` | -17 >> 3 = -3 (floor). Remainder = 7 (= -17 & 0x7). Threshold = 3 + 1 = 4. 7 > 4, round toward zero: -3 + 1 = -2. |
| `-16` | `3` | `-2` | -16 >> 3 = -2. Remainder = 0, no rounding. |
| `4` | `3` | `1` | 4 >> 3 = 0. Remainder = 4, threshold = 3. 4 > 3, round up to 1. |

<details><summary>Solution</summary>See `solutions/03-fusion/rdbpot.py` for a reference implementation.</details>

> **🔗 MLSys Connection:** These exact operations — SRDHM and RDBPOT — are defined in the [gemmlowp](https://github.com/google/gemmlowp) library that TFLite uses internally. They implement the fixed-point arithmetic needed for quantised inference. Every quantised model running on TFLite, whether on a phone CPU or an edge TPU, uses these same two primitives for requantisation. Understanding them is understanding the numerical foundation of INT8 inference.

---

## Exercise 3: Implement SRDHM in Hardware

Add SRDHM as a new CFU instruction at `funct3=1`.

**Spec:**
- Inputs: `in0` (a, INT32), `in1` (b, INT32)
- Output: `SRDHM(a, b)` as defined above
- Encoding: `funct3=1`, `funct7=0`

**Hardware considerations:**

The core of SRDHM is a 64-bit multiply (`int64(a) * int64(b)`). On the Tang Nano 20K's Gowin GW2AR-18C:

- DSP blocks are 18x18 signed multipliers
- A 32x32 multiply requires decomposition into multiple 18x18 multiplies
- Amaranth's `*` operator will synthesise this automatically — the tools will infer DSP usage
- Expected cost: 2-4 DSP blocks, depending on how the synthesiser packs them

**Exercise 3.1:** Create an `Srdhm` class that extends `Instruction` in `hardware/cfu.py`.

<details><summary>Hint 1</summary>

Start with the simplest possible implementation: compute `(int64(a) * int64(b) + (1 << 30)) >> 31` in combinational logic, handle the saturation case separately. Amaranth will handle the wide multiply.

</details>

<details><summary>Hint 2</summary>

In Amaranth, you can cast to wider signals using `.as_signed()` and `Cat(signal, Repl(signal[-1], N))` for sign extension. Or use `Signal(64)` and assign via combinational logic. The `>>` operator on signed signals does arithmetic right shift.

</details>

<details><summary>Hint 3</summary>

The saturation case: detect when both inputs equal `INT32_MIN` (i.e., bit 31 set, bits 30:0 all zero). Return `INT32_MAX` (0x7FFFFFFF) in that case. Use a `Mux` or `with m.If()`.

</details>

<details><summary>Solution</summary>See `solutions/03-fusion/srdhm_instruction.py` for the Amaranth implementation.</details>

---

## Exercise 4: Implement RDBPOT in Hardware

Add RDBPOT as a new CFU instruction at `funct3=2`.

**Spec:**
- Inputs: `in0` (a, INT32), `in1` (shift, treated as unsigned, range 0-31)
- Output: `RDBPOT(a, shift)` as defined above
- Encoding: `funct3=2`, `funct7=0`

**Hardware considerations:**

RDBPOT is cheaper than SRDHM — no DSP blocks needed. It's a barrel shifter (mux tree) plus some addition and comparison logic. Expected cost: ~50-80 LUTs.

**Exercise 4.1:** Create an `Rdbpot` class that extends `Instruction`.

<details><summary>Hint 1</summary>

The variable right shift is the main challenge. A barrel shifter for 32-bit values with 5-bit shift amount is a standard circuit: 5 stages of 2:1 muxes, each controlled by one bit of the shift amount.

</details>

<details><summary>Hint 2</summary>

In Amaranth, `a >> shift` with a `Signal` shift amount generates a barrel shifter. For signed arithmetic right shift, make sure `a` is `.as_signed()` before shifting.

</details>

<details><summary>Hint 3</summary>

Computing `remainder = a & mask` where `mask = (1 << shift) - 1`: use `(1 << shift)` with Amaranth's `shift_left` or `Cat(Const(1), Repl(0, shift))`. Alternatively, compute it as `a - (result << shift)` after the shift.

</details>

<details><summary>Solution</summary>See `solutions/03-fusion/rdbpot_instruction.py` for the Amaranth implementation.</details>

---

## Exercise 5: Wire Into the CFU and Test End-to-End

Now integrate both instructions into the full system.

**Exercise 5.1:** Register the new instructions in your `Cfu` top class:

```
funct3=0 → SimdMac4    (existing)
funct3=1 → Srdhm       (new)
funct3=2 → Rdbpot      (new)
```

**Exercise 5.2:** Add firmware wrappers in `cfu.zig`:

```
pub fn srdhm(a: i32, b: i32) i32    // funct3=1, funct7=0
pub fn rdbpot(a: i32, shift: i32) i32  // funct3=2, funct7=0
```

**Exercise 5.3:** Add protocol opcodes and dispatch handlers:

```
OP_SRDHM  = 0x02   in protocol.py
OP_RDBPOT = 0x03   in protocol.py

OpType.srdhm  = 0x02   in link.zig
OpType.rdbpot = 0x03   in link.zig
```

**Exercise 5.4:** Write end-to-end tests that exercise the full path:

1. Send SRDHM request from host, verify result matches Python reference
2. Send RDBPOT request from host, verify result matches Python reference
3. Send the full requantisation pipeline: MAC4 result -> SRDHM -> RDBPOT -> clamp
4. Verify against a known TFLite quantisation example

<details><summary>Hint 1</summary>

For the firmware dispatch handlers, follow the same pattern as `handle_mac4` in `dispatch.zig`: read 8 bytes of payload as two `i32` values, call the CFU function, send back 4 bytes.

</details>

<details><summary>Hint 2</summary>

For the end-to-end requantisation test, you need to chain three host calls (MAC4, SRDHM, RDBPOT) and verify the final INT8 output. The host drives the pipeline for now — true hardware fusion (single-command requant) comes in a later unit.

</details>

<details><summary>Solution</summary>See `solutions/03-fusion/test_requant.py` for end-to-end test code and `solutions/03-fusion/dispatch_updated.zig` for the firmware handlers.</details>

---

## Exercise 6: Measure the Fusion Benefit

**Exercise 6.1:** Time the unfused vs. fused requantisation pipeline.

```
UNFUSED (3 separate host round trips):
  MAC4 request + response:  ~2.4 ms
  SRDHM request + response: ~2.4 ms
  RDBPOT request + response: ~2.4 ms
  Python clamp:              ~0 ms
  Total:                     ~7.2 ms per output element

FUSED (if you had a single "requant" command):
  Single request + response: ~2.4 ms
  Total:                     ~2.4 ms per output element

  Speedup: 3×
```

This 3x improvement comes purely from eliminating round trips — no hardware change at all (same CFU, same clock speed, same UART baud rate).

**Exercise 6.2:** Now consider a layer producing 128 output elements. Unfused: 128 * 7.2 ms = 922 ms. Fused (single command per element): 128 * 2.4 ms = 307 ms. Fully autonomous (single command for all 128): ~2.4 ms + compute time. The autonomy improvement (Unit 4) dwarfs even fusion — but fusion is the prerequisite.

<details><summary>Hint 1</summary>

The lesson: fusion eliminates *per-element* overhead. Autonomy eliminates *per-operation* overhead. You need both. Fusion without autonomy still has per-element UART round trips. Autonomy without fusion still has intermediate result traffic. Together, they reduce communication to: one command to start, one bulk transfer of final results.

</details>

> **🔗 MLSys Connection:** This is the difference between XLA's *op fusion* (combining element-wise ops into a single kernel) and XLA's *buffer assignment* (keeping intermediate tensors in faster memory tiers). Op fusion reduces kernel launch overhead. Buffer assignment reduces memory traffic. Both are necessary. Your three exercises exactly mirror this: Exercise 5 implements op fusion (combining SRDHM + RDBPOT into the same hardware path), and Unit 4 will implement buffer assignment (keeping data in on-chip BSRAM instead of round-tripping through UART).

---

## Stretch: A True Fused Instruction

What if SRDHM and RDBPOT were a *single* fused CFU instruction?

```
funct3=3 → Requant: takes (accumulator, multiplier, shift) → INT8
```

**What would you save?**

- One fewer CFU call (1 instruction instead of 2)
- One fewer register read/write cycle
- The intermediate SRDHM result never materialises in a CPU register

**What would it cost?**

- A multi-cycle instruction (SRDHM's multiply + RDBPOT's shift + clamp)
- More complex control logic
- The `done` signal would need to deassert for 2-3 cycles while the pipeline fills

This is the *same tradeoff* as fusing conv + batchnorm + relu on a GPU: the fused kernel is more complex to write but eliminates intermediate memory traffic.

**Exercise 7.1 (stretch):** Sketch the Amaranth module for a fused `Requant` instruction. Don't implement it yet — just draw the datapath and count the pipeline stages.

<details><summary>Hint 1</summary>

You need 3 inputs (accumulator, multiplier, shift) but the CFU only provides 2 inputs per instruction. Options: (1) use `funct7` to encode the shift (only 5 bits needed, funct7 has 7 bits), (2) use a configuration register written by a prior instruction, (3) accept 2 cycles (first loads multiplier+shift, second triggers compute).

</details>

<details><summary>Hint 2</summary>

Look at how Google's `hps_accel` solves this: the PostProcess pipeline reads per-channel parameters (bias, multiplier, shift) from a dedicated BSRAM. The parameters are loaded once at layer setup time. The pipeline is fully autonomous — no per-element CPU involvement.

Reference: `docs/tutorial/appendix-prior-art.md`, Section A.3 (The Requantisation Pipeline).

</details>

---

## Checkpoint

Before moving to Unit 4, verify:

- [ ] You can compute SRDHM by hand and your Python reference matches all test vectors
- [ ] You can compute RDBPOT by hand and your Python reference matches all test vectors
- [ ] SRDHM hardware instruction works at `funct3=1` (passes all test vectors)
- [ ] RDBPOT hardware instruction works at `funct3=2` (passes all test vectors)
- [ ] End-to-end pipeline works: MAC4 -> SRDHM -> RDBPOT -> clamp -> correct INT8 output
- [ ] You can explain why fusion reduces latency by 3x without any hardware speedup
- [ ] You understand why SRDHM needs DSP blocks (64-bit multiply) but RDBPOT does not

**Full pipeline verification:**

```
  Input:  a = [1, 2, 3, 4], b = [1, 1, 1, 1], offset = 128
  MAC4:   (129 + 130 + 131 + 132) = 522
  SRDHM:  srdhm(522, 1073741824) = ?   (use your reference impl)
  RDBPOT: rdbpot(result, 2) = ?        (use your reference impl)
  Clamp:  max(-128, min(127, result))   (should be in INT8 range)
```

If the final INT8 value matches your Python reference, your fusion pipeline is correct.

---

## Suggested Readings

1. **Operator Fusion in ML Compilers:**
   - "XLA: Optimizing Compiler for Machine Learning" (Google, 2017) — describes the HLO fusion pass that combines element-wise operations into single kernels. Sections on "fusion" and "buffer assignment" are directly relevant.
   - "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning" (Chen et al., 2018) — Section 4.2 covers operator fusion. TVM classifies operators as injective, reduction, or complex, and fuses chains of injective ops into reductions.
   - "Rammer/NNFusion" (Ma et al., OSDI 2020) — goes beyond op fusion to *inter-operator* scheduling, fusing independent operators across branches of the computation graph.

2. **Quantised Inference Arithmetic:**
   - "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (Jacob et al., 2018) — the paper that defines the SRDHM/RDBPOT requantisation scheme used by TFLite. Section 2.2 derives the fixed-point math.
   - [gemmlowp documentation](https://github.com/google/gemmlowp/blob/master/doc/quantization.md) — the reference implementation of quantised matrix multiply, including the exact SRDHM and RDBPOT functions.

3. **Hardware Fusion in Accelerators:**
   - "In-Datacenter Performance Analysis of a Tensor Processing Unit" (Jouppi et al., 2017) — the TPU v1 paper. Section 4 describes how the activation pipeline (requantisation + nonlinearity) is fused with the systolic array output.
   - CFU-Playground `hps_accel` source code, `PostProcess` module — a concrete implementation of fused requantisation in Amaranth, directly applicable to your design. See `docs/tutorial/appendix-prior-art.md` Section A.3 for analysis.

4. **Memory Bandwidth as the Bottleneck:**
   - "Roofline: An Insightful Visual Performance Model" (Williams, Waterman, Patterson, 2009) — explains why low-arithmetic-intensity ops (like requantisation) are always memory-bound, and why fusion is the only way to improve them.
   - "Making Deep Learning Go Brrrr From First Principles" (Horace He, 2022) — accessible blog post explaining memory bandwidth bottlenecks in PyTorch, with concrete examples of why fusion matters.

---

**Previous:** [Unit 2 — The Data Path](02-datapath.md)
**Next:** [Unit 4 — Autonomous Compute: Taking the CPU Off the Hot Path](04-autonomy.md)
