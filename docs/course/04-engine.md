# Unit 4: The Execution Engine — GPU Cores on an FPGA

> **Series:** [01-numbers](01-numbers.md) → [02-datapath](02-datapath.md) → [03-vertical](03-vertical.md) → **[04-engine](04-engine.md)** → [05-compiler](05-compiler.md)

A GPU streaming multiprocessor (SM) receives a thread block, fetches instructions from memory, manages shared memory, and runs the compute loop — all without CPU involvement. You're going to build the same thing.

---

## 4.1  The Problem You're Solving

In earlier units, the CPU was a data pump — reading weights, feeding the MAC, reading results. Every cycle the CPU spends feeding data is a cycle the compute hardware sits idle:

```
  CPU as data pump (CSR writes):
  +-- Write operands:  ~3 cycles per CSR write x 2 writes = 6 cycles
  +-- Read result:     ~3 cycles
  +-- HW compute:      1 cycle
  +-- Utilization:     1 / 10 = 10%

  CPU as data pump (custom instructions):
  +-- Load operands:   ~1 cycle per instruction
  +-- Read result:     ~1 cycle
  +-- HW compute:      1 cycle
  +-- Utilization:     1 / 3 = 33%
```

The CPU is on the hot path. On a GPU, the host CPU submits work and walks away. The SM handles everything: instruction fetch, register file access, shared memory reads, compute, writeback. Your CPU should do the same — configure, launch, wait, read results.

> **MLSys Connection:** This is the fundamental difference between a CPU-attached coprocessor and an autonomous compute engine. In CUDA, `cuLaunchKernel()` returns immediately. The host is free. The GPU SM fetches its own instructions from memory, manages its own register file, and runs the compute loop independently. Your sequencer FSM is a stripped-down SM that runs a *fixed* instruction stream instead of a programmable one.

---

## 4.2  The GPU Analogy

Map your FPGA accelerator onto GPU concepts:

| GPU Concept | Your FPGA Analog | What It Does |
|---|---|---|
| Streaming Multiprocessor (SM) | Sequencer FSM + MAC + PostProcess | The autonomous compute unit |
| Shared memory (`__shared__`) | BSRAM stores | On-chip SRAM, software-managed |
| Kernel arguments | Config registers (N, K, quant params) | Per-launch parameters |
| `cuLaunchKernel()` | Write START signal | Initiates autonomous execution |
| Thread block dimensions | Compute descriptor (spatial, N, K) | Defines the work to execute |
| Warp scheduler | Sequencer FSM state transitions | Decides what to compute each cycle |
| Register file | Accumulator + pipeline registers | Per-element working state |

The key difference: a real SM has a *programmable* instruction stream fetched from memory. Your sequencer has a *fixed* inner loop (MAC over K, requant, output). This is a deliberate trade-off.

> **MLSys Connection:** Google's TPU makes the same trade-off. Its matrix multiply unit runs a fixed systolic dataflow — there is no instruction fetch for the inner loop. The "program" is the configuration (dimensions, addresses, quantization parameters), not a stream of ALU instructions. Specialization sacrifices generality for efficiency: fewer transistors on control, more on compute.

---

## 4.3  Key Questions

Before building anything, think through these. They connect your FPGA design to real GPU architecture.

**What is the "instruction stream"?**

Your sequencer has a fixed inner loop: iterate over K, MAC4, requant on last, write to FIFO, advance to next output channel. A real GPU SM fetches instructions from memory — loads, stores, FMAs, branches, barriers. What do you gain by fixing the loop in hardware? What do you lose? When is the trade-off worth it?

**What is "shared memory"?**

In CUDA, `__shared__` memory is a fast, software-managed scratchpad per thread block. Programmers explicitly load data from global memory into shared memory, then compute from it. Your BSRAM stores are the same thing — the CPU (host) loads data into BSRAM before compute, then the hardware reads from BSRAM during compute. The programmer (firmware) manages what goes where.

**What is "kernel launch"?**

In CUDA: `cuLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream)`. On your system: write config registers (dimensions, quant params) + write START signal. The conceptual mapping is exact — you are passing kernel arguments and initiating execution.

---

## 4.4  BSRAM Memory Map — Your Shared Memory Budget

The Tang Nano 20K (GW2AR-18C) has limited BSRAM. After the SoC claims its share, you have roughly 16 free BSRAMs. Each BSRAM is 18 Kbit = 2 KiB (1024x18-bit or 2048x9-bit words).

This is your shared memory budget. Every byte matters.

```
  BSRAM allocation plan:

  +----------------------------------------------+
  | Usage              BSRAMs    Capacity         |
  +----------------------------------------------+
  | Activation bank 0      1    2,048 bytes       |
  | Activation bank 1      1    2,048 bytes       |
  | Activation bank 2      1    2,048 bytes       |
  | Activation bank 3      1    2,048 bytes       |
  | Filter store 0         1    2,048 bytes       |
  | Filter store 1         1    2,048 bytes       |
  | Requant param store    2    ~3,456 bytes      |
  | Output FIFO            1    2,048 bytes       |
  +----------------------------------------------+
  | TOTAL                  9    ~18 KiB           |
  | Remaining            7-9    ~14-18 KiB free   |
  +----------------------------------------------+
```

**Activation banks (4 x 2 KiB = 8 KiB):** Phase-rotated for conflict-free access. On each cycle, four read ports access four different banks. The phase rotation ensures zero bank conflicts even with sequential addresses. This is the same trick GPU shared memory uses with bank interleaving.

**Filter stores (2 x 2 KiB = 4 KiB):** Weights pre-loaded before compute. A cyclic read pointer wraps around K — the hardware reads filter[0], filter[1], ..., filter[K/4-1], then wraps back to filter[0] for the next output channel. With 2 stores, you can double-buffer: load the next layer's weights while computing the current layer.

**Requant param store (2 BSRAMs):** Per-channel bias (18 bits), multiplier (32 bits), and shift (4 bits) = 54 bits per channel. For up to 512 output channels.

**Output FIFO (1 BSRAM = 2 KiB):** Packs 4 INT8 results per 32-bit word = 2048 output elements before full. The CPU drains it after DONE.

> **MLSys Connection:** CUDA shared memory on an A100 is 164 KiB per SM. Yours is ~18 KiB total. But the *architectural pattern* is identical: fast on-chip SRAM, software-managed, sized to hold the working set of the current computation. GPU kernel authors obsess over shared memory tile sizes — you face the exact same constraint. A 1x1 conv with 8 input channels and 4x4 spatial dims needs 128 bytes of activations. That fits trivially. A layer with 96x96 spatial and 8 channels needs 72 KiB — you must tile spatially, processing a few rows at a time. This is the same tiling problem that drives CUDA kernel design.

---

## 4.5  The Compute Descriptor — Your Kernel Arguments

Before the CPU hits START, it must configure the sequencer. These registers are your "kernel arguments" — the parameters that define *what* to compute.

**Exercise 4a: Design the compute descriptor.**

What registers does the CPU need to write before hitting START? Think about what the sequencer FSM needs to know to iterate over the full computation without any further CPU involvement.

Consider:
- How does the sequencer know when to stop accumulating and emit an output?
- How does it know how many output channels to produce?
- Where are the requantization parameters for each channel?
- Does it need spatial dimensions, or can it derive the iteration count?

<details><summary>Hint 1</summary>

The sequencer needs at minimum: the K dimension (input depth), the N dimension (output channels), and the number of spatial positions to process. These three values define the full iteration space.

</details>

<details><summary>Hint 2</summary>

Think about what Google's `hps_accel` uses: `input_depth`, `output_channels`, plus configuration for the `first`/`last` signal generation. The `first`/`last` pattern eliminates explicit K-tiling — the filter store read pointer wraps around every `K/4` cycles, and `first` resets the accumulator while `last` triggers output.

</details>

<details><summary>Hint 3</summary>

A minimal descriptor:
- `input_depth` (K): determines filter store wrap point and `first`/`last` generation
- `output_channels` (N): outer loop bound
- `num_spatial`: number of spatial positions (or total output count = spatial x N)
- `output_offset`: per-layer requant constant
- `activation_min`, `activation_max`: clamp bounds (usually -128, 127)

The per-channel params (bias, multiplier, shift) live in the param BSRAM, indexed by channel number.

</details>

<details><summary>Solution</summary>See solutions/04-unit/compute_descriptor.py</details>

---

## 4.6  BSRAM Filter and Activation Stores

The filter store holds weights. The activation store holds input data. Both are loaded by the CPU before compute and read autonomously by the hardware during compute.

### The Cyclic Read Pattern

The filter store uses a cyclic read pattern. For `input_depth=64` with 4 MACs per block, the store has `64/4 = 16` entries. The read pointer cycles through entries 0-15 repeatedly — once per output channel.

The `first`/`last` signals are derived from the pointer position:

```
  For input_depth=64 (K=64), 4 MACs per block:
  Filter store has 64/4 = 16 entries.

  Cycle:  0     1     2    ...   15    16    17   ...
  first:  1     0     0    ...    0     1     0   ...
  last:   0     0     0    ...    1     0     0   ...

  On first=1: accumulator resets to 0
  On last=1:  accumulator value goes to PostProcess pipeline
  Between:    accumulator keeps accumulating
```

The hardware processes the ENTIRE K dimension in one pass. No tiling, no partial-sum management, no CPU intervention.

### BSRAM Read Latency — The Prefetch Pattern

BSRAM is synchronous: you set a read address on one clock edge, the data appears on the next. This 1-cycle latency matters for the compute loop.

The solution is a prefetch pattern. Two hardware address counters (one per store) drive the SRAM read ports. The SRAM's internal output register acts as the holding buffer — no extra flip-flops needed:

```
  Cycle -1 (prime):  addr ← 0                        (no data yet)
  Cycle  0:          SRAM outputs data[0]             mac4_next uses it ✓
                     consume → addr ← 1
  Cycle  1:          SRAM outputs data[1]             mac4_next uses it ✓
                     consume → addr ← 2
  Cycle  2:          SRAM outputs data[2]             addi (loop overhead)
  Cycle  3:          SRAM still outputs data[2]       bne  (addr unchanged)
  Cycle  4:          SRAM still outputs data[2]       mac4_next uses it ✓
                     consume → addr ← 3
```

Between `mac4_next` calls, the SRAM output stays stable because the address doesn't change. During loop overhead (addi, bne), the prefetched data waits — free latency hiding.

**The two firmware rules:**
1. **Prime before first `mac4_next`** — 1 cycle gap so SRAM can present data[0]
2. **Write all data before prime** — no concurrent write + read

**Exercise 4b: Implement BSRAM stores with prefetch counters.**

The hardware you need for each store is minimal — an `amaranth.Memory` instance plus an address counter:

```python
# Filter store — ~15 lines of Amaranth
filt_mem  = Memory(shape=32, depth=512, init=[])
filt_rptr = Signal(16)   # read pointer: auto-inc on consume, wraps at k_count
k_count   = Signal(16)   # wrap point, set via CFU instruction
first     = Signal()     # filt_rptr == 0
last      = Signal()     # filt_rptr == k_count - 1

# Activation store — same pattern
act_mem   = Memory(shape=32, depth=512, init=[])
act_rptr  = Signal(16)   # read pointer: auto-inc on consume, firmware resets between channels
```

Write pointers live in firmware (the CPU passes explicit addresses). Read pointers and `k_count` live in hardware (5 signals + 2 comparators total).

<details><summary>Hint 1</summary>

Start with the filter store alone. Use Amaranth's `Memory` with one write port (driven by the `WriteRegs` instruction) and one read port (address driven by `filt_rptr`). The counter logic:

```python
# On consume (mac4_next fires):
with m.If(consume):
    with m.If(filt_rptr == k_count - 1):
        m.d.sync += filt_rptr.eq(0)       # cyclic wrap
    with m.Else():
        m.d.sync += filt_rptr.eq(filt_rptr + 1)

m.d.comb += first.eq(filt_rptr == 0)
m.d.comb += last.eq(filt_rptr == k_count - 1)
```

</details>

<details><summary>Hint 2</summary>

Wire the SRAM read data directly to the MAC operand mux — no holding register needed. The SRAM's internal output register IS the buffer. The `mac4_next` path (funct7=1) selects SRAM output instead of CPU `in0`/`in1`:

```python
m.d.comb += [
    mac.in0.eq(Mux(is_mac4_next, act_mem_rd_data, self.cmd_in0)),
    mac.in1.eq(Mux(is_mac4_next, filt_mem_rd_data, self.cmd_in1)),
]
```

</details>

<details><summary>Hint 3</summary>

For the activation store, start with a single BSRAM and sequential reads. Phase-rotated multi-bank access is an optimization for later. Get single-bank working first.

</details>

<details><summary>Solution</summary>See solutions/04-engine/bsram_stores.md</details>

---

## 4.7  The Sequencer — Who Runs the Loop?

The core design question: something must iterate over all `spatial x N x K/4` MAC operations. You have two choices.

### Option A: Hardware FSM Sequencer

A dedicated FSM in Amaranth runs the triple-nested loop autonomously. The CPU writes config, asserts START, and waits for DONE. The MAC runs every cycle — no CPU involvement.

### Option B: CPU-as-Sequencer (Software-Unrolled)

The VexRiscv CPU runs the loop in firmware. The existing `SimdMac4` is unchanged — a `funct7=1` variant (`mac4_next`) muxes its operands from BSRAM read ports instead of CPU registers. Two hardware address counters auto-increment on each `mac4_next`, with the filter counter wrapping cyclically at `k_count`.

The CFU instruction ABI extends the existing `WriteRegs`/`ReadRegs` pattern using `funct7` to sub-select registers:

```
  funct3  funct7  Name                  in0         in1
  ──────  ──────  ────                  ───         ───
    0       0     MAC4                  activations weights      (existing)
    0       1     MAC4_NEXT             (ignored)   (ignored)    (reads from BSRAM)

    1       0     READ input_offset                              → value
    1       1     READ accumulator                               → value

    2       0     WRITE input_offset                in1=value
    2       1     RESET accumulator
    2       2     WRITE k_count                     in1=value
    2       3     WRITE filter SRAM     in0=addr    in1=data
    2       4     WRITE act SRAM        in0=addr    in1=data
    2       5     WRITE param SRAM      in0=addr    in1=data
    2       6     RESET filter read ptr (prime)
    2       7     RESET act read ptr

    3       0     SRDHM                 in0=a       in1=b        (existing)
    4       0     RDBPOT                in0=x       in1=shift    (existing)
```

The firmware loop:

```zig
fn computeLayer(k_count: u16, n_channels: u16, spatial: u16) void {
    cfu.setKCount(k_count);

    for (0..spatial) |_| {
        cfu.resetFilterRPtr();    // prime: resets to addr 0, SRAM presents data[0] next cycle
        for (0..n_channels) |n| {
            cfu.resetAcc();
            cfu.resetActRPtr();   // same activations reused per channel

            // Inner K loop — unrolled, back-to-back mac4_next works after prime
            var k: u16 = 0;
            while (k + 4 <= k_count) : (k += 4) {
                _ = cfu.mac4Next();
                _ = cfu.mac4Next();
                _ = cfu.mac4Next();
                _ = cfu.mac4Next();
            }
            while (k < k_count) : (k += 1) _ = cfu.mac4Next();

            const acc = cfu.readAcc();
            output[n] = requant(acc, params[n]);  // CPU-driven via SRDHM + RDBPOT
        }
    }
}
```

### Cycle-by-Cycle Comparison

For K=8 (2 MAC4 ops per channel), N=4 output channels, S=1 spatial. Total useful MAC4 ops = **8**.

**Hardware FSM** — MAC4 every cycle, postprocess overlaps with next channel:

```
  Cycle:  0     1     2     3     4     5     6     7     8     9
          ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
  MAC4:   │fill │ M0  │ M1  │ M2  │ M3  │ M4  │ M5  │ M6  │ M7  │
          │     │ ch0 │ ch0 │ ch1 │ ch1 │ ch2 │ ch2 │ ch3 │ ch3 │
          ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
  PP:     │     │     │     │ PP0 │     │ PP1 │     │ PP2 │     │←PP3
          └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

  MAC active: 8 / 10 = 80%
```

**CPU-driven, inner K unrolled** — MAC4 runs when CPU issues `mac4_next`, idle during loop overhead:

```
  Cycle:  0     1     2     3     4     5     6     7     8     9    10    11
          ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
  CPU:    │ mac │ mac │ addi│ bne │ mac │ mac │ addi│ bne │ mac │ mac │ ... │
          │ _nxt│ _nxt│ cnt │ loop│ _nxt│ _nxt│ cnt │ loop│ _nxt│ _nxt│     │
          │ ch0 │ ch0 │     │     │ ch1 │ ch1 │     │     │ ch2 │ ch2 │     │
          ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
  MAC4:   │ ██  │ ██  │     │     │ ██  │ ██  │     │     │ ██  │ ██  │     │
          └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                        ▲▲                ▲▲
                        └┴── loop overhead: MAC idle

  MAC active: 8 / 12 = 67%
```

**CPU-driven, inner K NOT unrolled** — branch overhead on every MAC4:

```
  Cycle:  0     1     2     3     4     5     6     7     ...
          ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
  CPU:    │ mac │ addi│ bne │ mac │ addi│ bne │ mac │ addi│
          │ _nxt│ k-- │k_lp │ _nxt│ k-- │n_lp │ _nxt│ k-- │
          ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
  MAC4:   │ ██  │     │     │ ██  │     │     │ ██  │     │
          └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

  MAC active: 1 per 3 cycles = 33%
```

### Utilization Scaling with K

| K (input depth) | K/4 (MACs/ch) | HW FSM | CPU unrolled | CPU loop |
|---|---|---|---|---|
| 8 | 2 | 80% | 67% | 33% |
| 16 | 4 | 89% | 80% | 33% |
| 32 | 8 | 94% | 89% | 33% |
| 64 | 16 | 97% | 94% | 33% |

```
  HW FSM:          MAC_cycles / (MAC_cycles + 1 + PP_drain)
  CPU unrolled:    MAC_cycles / (MAC_cycles + 2 × N_channels)
  CPU loop:        MAC_cycles / (MAC_cycles × 3)  →  always 33%
```

The gap between HW FSM and CPU-unrolled shrinks as K grows. For MobileNet layers (K=32–96), CPU-unrolled gives **89–97%** — nearly matching the hardware FSM with far less design complexity.

### The Design Decision

The CPU-as-sequencer approach is the right first step:

1. **Simpler hardware.** No triple-nested counter FSM, no pipeline hazard tracking. The BSRAM stores and `mac4_next` instruction are the only new hardware.
2. **Debuggable.** The firmware loop is readable Zig code. You can print intermediate values, add breakpoints, test one channel at a time.
3. **Good enough.** For K≥16 (most real layers), CPU-unrolled achieves >80% MAC utilization.
4. **Upgrade path.** The hardware FSM is a drop-in replacement for the firmware loop — same stores, same MAC, same postprocess. Build it as a stretch goal when you need the last 10%.

> **MLSys Connection:** This mirrors the real GPU evolution. Early GPUs had fixed-function pipelines configured by the CPU. Programmable shaders came later, and only when the fixed-function approach couldn't express the needed operations. Google's TPU v1 also uses a host-configured sequencer — the host CPU sets up the systolic array configuration, then the hardware runs. Your CPU-as-sequencer is the same pattern: the CPU configures and drives the control flow, the hardware does the compute.

**Exercise 4c: Implement `mac4_next` and the firmware sequencer.**

Build a `mac4_next` CFU instruction that:
- Reads the current word from the filter BSRAM (pre-staged address)
- Reads the current word from the activation BSRAM
- Feeds both to the existing SimdMac4
- Auto-increments both read pointers
- Generates `first`/`last` from the filter store's cyclic counter position

Then write a firmware loop (in Zig or pseudocode) that uses `mac4_next` to compute a full 1x1 conv layer with K-unrolling.

<details><summary>Hint 1</summary>

The `mac4_next` instruction is the composition of your existing `SimdMac4` with the BSRAM stores. It doesn't need `in0`/`in1` from the CPU — it reads operands directly from the stores. The CPU just issues the instruction; the hardware does the rest. The `first` signal resets the accumulator, the `last` signal means "this channel is done."

</details>

<details><summary>Hint 2</summary>

For the firmware K-unrolling: if `K/4` is known at compile time (or at least bounded), you can use Zig's `comptime` to fully unroll the inner loop, eliminating all branch overhead. For variable K, unroll by 4 or 8 and handle the remainder.

```zig
// Unrolled by 4:
var k: u16 = 0;
while (k + 4 <= k_count) : (k += 4) {
    _ = cfu.mac4_next();
    _ = cfu.mac4_next();
    _ = cfu.mac4_next();
    _ = cfu.mac4_next();
}
while (k < k_count) : (k += 1) {
    _ = cfu.mac4_next();  // remainder
}
```

</details>

<details><summary>Hint 3</summary>

The activation store read pointer must reset at the start of each output channel (same activations, different weights). The filter store pointer keeps advancing (different weights per channel). Add a `reset_act_ptr` signal or CFU instruction that the firmware calls between channels.

</details>

<details><summary>Solution</summary>See solutions/04-unit/sequencer.py</details>

---

## 4.8  The Output FIFO

The output FIFO decouples compute from result delivery. Without it, the hardware stalls every time it produces an output (waiting for the CPU to read). With it, the hardware runs continuously and the CPU drains results at leisure after DONE.

```
  Without FIFO:                    With FIFO:

  HW: compute --> CPU: read        HW: compute --> FIFO --> CPU: drain
  HW: wait...    CPU: process      HW: compute --> FIFO       (later)
  HW: compute --> CPU: read        HW: compute --> FIFO
  HW: wait...    CPU: process      HW: compute --> FIFO

  Array stalls every output.       Array runs continuously.
  CPU and HW take turns.           CPU and HW are decoupled.
```

**Exercise 4d: Implement the output FIFO.**

Use 1 BSRAM as a circular buffer. The hardware writes INT8 results (packed 4 per 32-bit word) during compute. The CPU reads 32-bit words after DONE.

You need:
- A write pointer (hardware side, advances during compute)
- A read pointer (CPU side, advances on each read)
- A count or empty/full flag
- An interface for the CPU to read: a CFU instruction that returns the next FIFO word and advances the read pointer

<details><summary>Hint 1</summary>

A BSRAM-backed FIFO is just two pointers into a memory. Write pointer increments when the PostProcess pipeline emits a result. Read pointer increments when the CPU issues a "read FIFO" instruction. The FIFO is empty when `read_ptr == write_ptr` and full when `write_ptr - read_ptr == depth`.

</details>

<details><summary>Hint 2</summary>

For INT8 packing: the PostProcess pipeline emits one INT8 result at a time. Accumulate 4 results into a 32-bit packing register, then write one 32-bit word to BSRAM. This 4:1 packing means 2048 bytes of BSRAM holds 2048 INT8 outputs.

</details>

<details><summary>Solution</summary>See solutions/04-unit/output_fifo.py</details>

---

## 4.9  Utilization Analysis

Now measure what you've built. The whole point of BSRAM stores + `mac4_next` is utilization — what fraction of cycles is the MAC actually computing?

**Exercise 4e: Measure utilization.**

For a 1x1 conv layer with 8 input channels, 8 output channels, and 4x4 spatial:

**Unit 2 baseline (CPU as data pump, per-element CFU calls):**
- 16 spatial × 8 channels × 2 MAC4 invocations × ~3 cycles (CFU overhead) = ~768 cycles
- MAC active cycles: 16 × 8 × 2 = 256
- Utilization: 256 / 768 = **33%**

**Unit 4 target (BSRAM stores + `mac4_next`, K-unrolled):**
- Load weights: 8 × 8 = 64 bytes into BSRAM = ~64 CPU cycles
- Load activations: 16 × 8 = 128 bytes = ~128 CPU cycles
- Compute (firmware loop, K=8 unrolled):
  - Per output element: 2 × `mac4_next` + 2 cycles outer loop = 4 cycles
  - 16 × 8 = 128 output elements × 4 = 512 cycles
  - MAC active: 256 of 512 = **67% during compute**
- Drain output: 128 INT8 results = ~128 CPU cycles
- Total: 64 + 128 + 512 + 128 = 832 cycles
- MAC active: 256 / 832 = **31% end-to-end** (transfer-bound)
- MAC active: 256 / 512 = **67% during compute** (compute-bound)

**With deeper K-unrolling or larger K:**

| Layer (S×N×K) | MAC ops | Compute cycles | Compute util | End-to-end util |
|---|---|---|---|---|
| 16×8×8 (tiny) | 256 | 512 | 67% | 31% |
| 16×8×32 | 1024 | 1280 | 80% | 69% |
| 16×32×32 | 4096 | 4608 | 89% | 84% |
| 64×32×64 | 32768 | 34816 | 94% | 93% |

The pattern: larger layers are compute-dominated (transfer overhead amortizes away). Smaller layers are transfer-dominated. This is the same roofline behavior as GPU kernels — small kernels are launch-overhead-bound, large kernels are compute-bound.

> **MLSys Connection:** GPU kernel performance is measured in "occupancy" and "compute utilization." A kernel that achieves 80%+ SM utilization is considered well-optimized. Memory-bound kernels often see 30–50%. Your CPU-as-sequencer with K-unrolling achieves 67–94% MAC utilization during compute, depending on K depth. The remaining overhead is firmware loop management — the same overhead that GPU warp schedulers eliminate in hardware. The upgrade path to a hardware FSM sequencer (Section 4.12) eliminates this last gap.

---

## 4.10  What Changes in the Firmware

The firmware simplifies dramatically:

```
  Unit 2 firmware:                   Unit 4 firmware:
  -------------------------          --------------------------
  receive data byte by byte          receive data in bulk
  loop calling cfu.mac4()            load BSRAM via writeFilter/writeAct
  track accumulator in software      prime prefetch, run K-unrolled loop
  send result                        requant via SRDHM + RDBPOT
                                     send results
```

The protocol changes too. Instead of streaming operands with each request, the host sends weight and activation blobs that get loaded into BSRAM. Then an "execute" command triggers the firmware compute loop. Then the host reads back results.

```
  OpType:
    load_weights = 0x01    // bulk-load weight data into filter BSRAM
    load_acts    = 0x02    // bulk-load activation data into act BSRAM
    load_params  = 0x03    // load requant params into param BSRAM
    configure    = 0x04    // set K, N, spatial, offsets
    execute      = 0x05    // firmware runs compute loop, returns results
    ping         = 0x06    // health check
```

The `cfu.zig` wrappers map directly to the funct3/funct7 ABI:

```zig
// funct3=0: compute
pub inline fn mac4(a: i32, b: i32) i32 { return cfu_op(0, 0, a, b); }
pub inline fn mac4Next() i32 { return cfu_op(0, 1, 0, 0); }

// funct3=1: reads
pub inline fn readAcc() i32 { return cfu_op(1, 1, 0, 0); }

// funct3=2: writes (funct7 selects register)
pub inline fn setInputOffset(v: i32) void { _ = cfu_op(2, 0, 0, v); }
pub inline fn resetAcc() void { _ = cfu_op(2, 1, 0, 0); }
pub inline fn setKCount(v: i32) void { _ = cfu_op(2, 2, 0, v); }
pub inline fn writeFilter(addr: i32, data: i32) void { _ = cfu_op(2, 3, addr, data); }
pub inline fn writeAct(addr: i32, data: i32) void { _ = cfu_op(2, 4, addr, data); }
pub inline fn writeParam(addr: i32, data: i32) void { _ = cfu_op(2, 5, addr, data); }
pub inline fn resetFilterRPtr() void { _ = cfu_op(2, 6, 0, 0); }
pub inline fn resetActRPtr() void { _ = cfu_op(2, 7, 0, 0); }

// funct3=3,4: requant (existing)
pub inline fn srdhm(a: i32, b: i32) i32 { return cfu_op(3, 0, a, b); }
pub inline fn rdbpot(x: i32, shift: i32) i32 { return cfu_op(4, 0, x, shift); }
```

Compare this to CUDA's kernel launch sequence: `cuMemcpyHtoD()` (load data), `cuLaunchKernel()` (execute), `cuMemcpyDtoH()` (read results). The structure is identical.

---

## 4.11  Checkpoint

After completing this unit, you should have:

- [ ] BSRAM filter store with auto-incrementing read, cyclic wrap, and `first`/`last` generation
- [ ] BSRAM activation store with auto-incrementing read and pointer reset
- [ ] A `mac4_next` CFU instruction that reads both stores, MACs, and auto-increments
- [ ] Firmware loop with K-unrolling that drives the compute
- [ ] An output buffer (FIFO or direct) that the CPU reads after compute
- [ ] Correct INT8 output for a known matmul (compare against NumPy reference)
- [ ] Measured MAC utilization ≥ 67% during compute phase

**The test:** Firmware loads weights + activations into BSRAM, runs the K-unrolled loop calling `mac4_next`, requants each output channel, stores results. Output matches NumPy reference for a known 1x1 conv (K=8, N=8, S=4×4).

---

## 4.12  Stretch Goals — Smarter Scheduling

The basic sequencer is single-shot: load → compute → drain → done. Real GPU schedulers are far more sophisticated. Once your basic sequencer works, consider these extensions — each maps to a real GPU/TPU scheduling concept.

### Double-Buffered Ping-Pong

Overlap loading the *next* tile's data with computing the *current* tile.

```
Time:    ──────────────────────────────────────────►
Compute: [  tile 0  ] [  tile 1  ] [  tile 2  ]
Load:        [ tile 1 load ] [ tile 2 load ] [ tile 3 ]
```

You already have 2 filter store BSRAMs allocated. While the sequencer reads from store A, the CPU writes the next tile into store B. On completion, swap roles. Just a 1-bit "active store" selector and a READY signal telling the CPU it can start loading the other bank.

> **🔗 MLSys Connection:** CUDA double-buffered shared memory tiling. Kernels load the next tile into shared memory while computing the current tile, hiding memory latency behind compute. `__syncthreads()` between phases.

### Command Queue (Ring Buffer)

CPU queues multiple compute descriptors. Hardware processes them back-to-back without CPU round trips between tiles.

```
CPU:  [push desc0][push desc1][push desc2]  ...  [wait DONE] [drain FIFO]
HW:               [compute 0][compute 1][compute 2]
```

A small ring buffer (4–8 entries) of descriptors in registers. The sequencer pops the next descriptor when it finishes the current one. The CPU fills the queue ahead of time.

This is the single biggest latency win for multi-tile layers — eliminates per-tile round trips entirely.

> **🔗 MLSys Connection:** This is exactly a GPU command buffer / CUDA stream. The CPU submits a sequence of kernel launches, the GPU processes them in order. `cudaStreamSynchronize()` = your "wait for queue empty" signal.

### Async Load Engine (DMA-like)

Separate "load," "compute," and "drain" into three independent FSMs that coordinate via semaphores.

```
Load FSM:    [load tile 0] [load tile 1] [load tile 2] ...
Compute FSM:      [wait]   [compute 0]   [compute 1]  ...
Drain FSM:                      [wait]   [drain 0]    ...
```

Maximum overlap — all three phases run concurrently. Approaches the theoretical throughput limit. But three interacting FSMs with synchronization is significantly harder to debug.

> **🔗 MLSys Connection:** The GPU's copy engine + compute engine + DMA engine running concurrently. In CUDA, `cudaMemcpyAsync` on stream 1 overlaps with kernel execution on stream 2.

### Layer-Fused Pipeline

Don't write intermediate results back to host between consecutive layers. Keep activations on-chip.

```
Layer 1 output → stays in activation BSRAM → becomes Layer 2 input
```

After finishing layer N, swap output FIFO ↔ activation bank roles. Only works when intermediate activations fit in BSRAM (~2 KiB), but for small spatial dims this eliminates host round trips between layers entirely. Needs the command queue as a prerequisite.

> **🔗 MLSys Connection:** XLA/torch.compile inter-op fusion. Instead of `kernel1 → write DRAM → kernel2 → read DRAM`, you get `kernel1 → registers/shared_mem → kernel2`. The "fusion" eliminates the memory round trip.

### Tiled 2D Scheduler

For layers where activations exceed BSRAM capacity, tile along *both* spatial and channel dimensions. The scheduler walks a 2D tile grid.

```
Output tensor:
  ┌──────┬──────┬──────┐
  │tile00│tile01│tile02│  ← spatial tiles (rows)
  ├──────┼──────┼──────┤
  │tile10│tile11│tile12│  ← channel tiles (columns)
  └──────┴──────┴──────┘
```

K-tiling (splitting the input channel dimension across multiple passes) requires partial sum management: "first K-tile" resets the accumulator, "middle K-tiles" accumulate, "last K-tile" triggers requant + output. Required for MobileNet's larger layers.

> **🔗 MLSys Connection:** How CUDA matmul kernels tile: thread blocks cover a 2D grid of output tiles, each block loads its tile of A and B into shared memory. The CUTLASS library's tile iterators are a production version of this.

### Hardware FSM Sequencer

Replace the firmware loop with a hardware FSM that runs the triple-nested loop (spatial × N × K/4) autonomously. The CPU writes config registers, asserts START, and waits for DONE.

```
  FSM state diagram:

  IDLE --- (CPU writes START) ---> RUNNING
                                     |
                                     | for each spatial position:
                                     |   for each output channel:
                                     |     for each K/4 chunk: MAC4
                                     |     on last: requant → FIFO write
                                     |
                                     v
  IDLE <--- (all outputs done) --- DONE
```

This eliminates all firmware loop overhead — the MAC runs every cycle during RUNNING. Utilization goes from 67–94% (CPU-driven) to 80–97% (hardware FSM). The gain is largest for small K values where loop overhead dominates.

The stores, MAC, and postprocess pipeline are unchanged — the FSM is a drop-in replacement for the firmware loop. You need three hardware counters (`k_counter`, `n_counter`, `spatial_counter`) and address generation logic for each store.

> **🔗 MLSys Connection:** This is the GPU warp scheduler — hardware that decides what to execute each cycle, eliminating software loop overhead entirely. The tradeoff: more hardware complexity for higher utilization. Worth it when the workload justifies the design effort.

### Recommended order

1. **Hardware FSM sequencer** — drop-in replacement for firmware loop. Biggest utilization gain for small K layers.
2. **Command queue** — biggest bang for complexity. Eliminates per-tile CPU round trips.
3. **Double-buffer** — nearly free given existing 2 filter BSRAMs. Layer on top of command queue.
4. **2D tiling** — needed when MobileNet layers don't fit. Orthogonal to the above.
5. **Async load / layer fusion** — powerful but high complexity. Pursue if you want to push utilization toward theoretical limits.

---

## Side Quests

- **Hardware profiling counters.** Add CSR-readable cycle counters that measure: (a) cycles spent in compute, (b) cycles waiting for data, (c) cycles idle. Three counters, ~50 LUTs. This tells you *exactly* where utilization is lost — the numbers will surprise you.
- **Interrupt-driven output.** Instead of polling the output FIFO, fire a RISC-V interrupt when the FIFO is half-full. The CPU can prefetch the next layer's config while waiting. LiteX's `EventManager` makes this straightforward.
- **DMA engine.** Build a small DMA controller that copies from `main_ram` to BSRAM. The CPU writes source address, destination, and length into CSRs, then the DMA does the transfer autonomously. This is how real GPUs feed their SMs — and it's ~200 lines of Amaranth.
- **PSRAM prefetch.** Connect the Tang Nano 20K's on-package 8 MiB PSRAM via LiteX's `litehyperbus`. Store the full model's weights in PSRAM, load into filter BSRAM per-layer. This removes the "weights must fit in BSRAM" constraint entirely.
- **Double-buffer activations.** Use two activation BSRAM banks as a ping-pong pair: the sequencer reads from bank A while the CPU (or DMA) loads bank B. When a tile finishes, swap. This overlaps data loading with compute.

---

## Suggested Readings

1. **NVIDIA CUDA Programming Guide, Chapter 4: Hardware Implementation** — How SMs, warp schedulers, and shared memory actually work. The concepts map directly to what you built.
   [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation)

2. **Google TPU v1 paper (Jouppi et al., 2017)** — "In-Datacenter Performance Analysis of a Tensor Processing Unit." The TPU's systolic array uses fixed dataflow, just like your sequencer. Section 3 on the TPU architecture is most relevant.
   [https://arxiv.org/abs/1704.04760](https://arxiv.org/abs/1704.04760)

3. **CFU-Playground paper (Prakash et al., 2023)** — The project this accelerator is based on. Study the `mnv2_first` and `hps_accel` examples for the autonomous sequencer and BSRAM store patterns.
   [https://github.com/google/CFU-Playground](https://github.com/google/CFU-Playground)

4. **Eyeriss: An Energy-Efficient Reconfigurable Accelerator (Chen et al., 2016)** — Detailed analysis of dataflow choices (row-stationary, output-stationary, weight-stationary) and their energy implications. Helps you understand why your BSRAM allocation matters.
   [https://eyeriss.mit.edu/](https://eyeriss.mit.edu/)

5. **Efficient Processing of Deep Neural Networks (Sze et al., 2020)** — Textbook-length tutorial covering the full design space: dataflow, memory hierarchy, hardware mapping. Chapters 6-7 on dataflow and energy-efficient design are most relevant to this unit.
   [https://arxiv.org/abs/2104.10462](https://arxiv.org/abs/2104.10462)

6. **FIFO and DMA Design:**
   - Cummings, ["Simulation and Synthesis Techniques for Asynchronous FIFO Design"](http://www.sunburst-design.com/papers/CummingsSNUG2002SJ_FIFO1.pdf) (SNUG 2002) — the definitive paper on async FIFO design. Even though your FIFO is synchronous, the concepts (gray-code pointers, full/empty detection) apply.
   - Patterson & Hennessy, *Computer Organization and Design: RISC-V Edition* — Ch. 5 on memory hierarchy. The DMA engine you might build for the PSRAM side quest is a simplified version of what this chapter describes.

---

**Previous:** [Unit 3 — Vertical Slice](03-vertical.md)
**Next:** [Unit 5 — The Compiler](05-compiler.md)
