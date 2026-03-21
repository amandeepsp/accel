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

**Exercise 4b: Implement BSRAM stores for weights and activations.**

Build the BSRAM stores in Amaranth. The filter store needs:
- A write port (CPU loads weights before compute)
- A read port with an auto-incrementing address counter
- Cyclic wrap: address resets to 0 after reaching `K/4 - 1`
- `first` and `last` output signals derived from the address counter

The activation store needs:
- A write port (CPU loads activations before compute)
- A read port with a sequential address counter
- 4 banks, phase-rotated for conflict-free access

<details><summary>Hint 1</summary>

Start with the filter store alone — it's simpler. Use Amaranth's `Memory` (or Gowin BSRAM primitive) with one write port and one read port. The key is the address counter: it increments every cycle during RUNNING, and wraps when it hits `k_count - 1`.

</details>

<details><summary>Hint 2</summary>

The `first` signal is `address_counter == 0`. The `last` signal is `address_counter == k_count - 1`. These feed directly into the MAC unit to control accumulator reset and output latching. Study how Google's `mnv2_first` generates these from `FilterStore`.

</details>

<details><summary>Hint 3</summary>

For the activation banks, you don't need full phase rotation in your first version. Start with a single activation BSRAM and sequential reads. Phase-rotated multi-bank access is an optimization for when you need higher read bandwidth (multiple spatial positions per cycle). Get single-bank working first, then generalize.

</details>

<details><summary>Solution</summary>See solutions/04-unit/bsram_stores.py</details>

---

## 4.7  The Sequencer FSM — Your Warp Scheduler

The sequencer is the heart of autonomous compute. It replaces the CPU's inner loop with a hardware FSM that iterates over all `spatial x N x K/4` MAC operations.

```
  FSM state diagram:

  IDLE --- (CPU writes START) ---> RUNNING
                                     |
                                     | (for each spatial position)
                                     |   (for each output channel)
                                     |     (for each K/4 chunk: MAC4)
                                     |     (on last: requant -> FIFO write)
                                     |
                                     v
  IDLE <--- (all outputs done) --- DONE
```

During RUNNING, the sequencer:
1. Reads filter data from filter BSRAM (cyclic, auto-incrementing)
2. Reads activation data from activation BSRAM (sequential)
3. Issues MAC4 to the compute pipeline
4. On `last` (end of K dimension): triggers PostProcess (requant), writes INT8 result to output FIFO
5. Advances to next output channel (resets activation read pointer, continues filter pointer)
6. After all channels for all spatial positions: asserts DONE

**Exercise 4c: Implement the sequencer FSM.**

Build the three-state FSM (IDLE, RUNNING, DONE) that controls the autonomous compute loop. The RUNNING state must manage three nested counters: spatial position, output channel, and K-chunk.

<details><summary>Hint 1</summary>

Start with the counter logic. You need three counters:
- `k_counter`: 0 to `K/4 - 1`, innermost loop
- `n_counter`: 0 to `N - 1`, middle loop
- `spatial_counter`: 0 to `num_spatial - 1`, outermost loop

On each cycle in RUNNING: increment `k_counter`. When it wraps, increment `n_counter`. When `n_counter` wraps, increment `spatial_counter`. When `spatial_counter` wraps, transition to DONE.

</details>

<details><summary>Hint 2</summary>

The `first` and `last` signals come from `k_counter`:
- `first = (k_counter == 0)` — tells the MAC to reset its accumulator
- `last = (k_counter == k_limit - 1)` — tells the pipeline to emit a result

Connect `first` to the MAC's reset input (your existing `funct3[0]` reset mechanism). Connect `last` to the PostProcess pipeline's "latch output" signal.

</details>

<details><summary>Hint 3</summary>

The address generation logic:
- Filter store address = `n_counter * k_limit + k_counter` (or equivalently, a single counter that wraps every `k_limit` cycles — the cyclic pattern)
- Activation address = `spatial_counter * k_limit + k_counter` (resets to the same spatial position's data for each output channel)
- Param store address = `n_counter` (one set of requant params per output channel)
- Output FIFO write = triggered on `last` after PostProcess pipeline latency

</details>

<details><summary>Hint 4</summary>

Be careful with pipeline latency. If your PostProcess pipeline takes P cycles, the output FIFO write happens P cycles after `last` asserts. You need a shift register or delayed signal to track this. The sequencer should not transition to DONE until the last result has been written to the FIFO.

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

Now measure what you've built. The whole point of autonomous compute is utilization — what fraction of cycles is the MAC actually computing?

**Exercise 4e: Measure utilization.**

For a 1x1 conv layer with 8 input channels, 8 output channels, and 4x4 spatial:

**Part 2 (CPU as data pump):**
- 16 spatial positions x 8 output channels x 2 MAC4 invocations x ~10 cycles (CSR overhead) = ~2560 cycles
- MAC active cycles: 16 x 8 x 2 = 256
- Utilization: 256 / 2560 = **10%**

**Part 3 target (autonomous with BSRAM):**
- Load weights: 8 x 8 = 64 bytes into BSRAM via CSR writes = ~192 CPU cycles
- Load activations: 16 x 8 = 128 bytes = ~384 CPU cycles
- Compute (autonomous): 16 x 8 x 2 = 256 MAC cycles, pipelined
- Drain FIFO: ~32 CPU cycles (128 INT8 outputs / 4 per word = 32 reads)
- MAC active cycles during compute phase: 256
- Total compute phase cycles: ~256 (MAC runs every cycle)
- Utilization during compute: **~100%**

The MAC never stalls during the autonomous compute phase. All the overhead moves to setup and teardown, which happen once per layer, not once per element.

> **MLSys Connection:** GPU kernel performance is measured in "occupancy" and "compute utilization." A kernel that achieves 80%+ SM utilization is considered well-optimized. Memory-bound kernels often see 30-50%. Your autonomous sequencer eliminates the control overhead that dominated the CPU-driven version, achieving near-perfect compute utilization for the actual compute phase. The remaining bottleneck is data loading — exactly the same situation as GPU kernels bottlenecked on global memory bandwidth.

---

## 4.10  What Changes in the Firmware

The firmware simplifies dramatically:

```
  Unit 2 firmware:                   Unit 4 firmware:
  -------------------------          --------------------------
  receive data byte by byte          receive data in bulk
  loop calling cfu.mac4()            load BSRAM (burst write)
  track accumulator in software      configure sequencer registers
  send result                        write START
                                     wait for DONE
                                     drain output FIFO
                                     send results
```

The protocol changes too. Instead of streaming operands with each request, the host sends weight and activation blobs that get loaded into BSRAM. Then a short "execute" command triggers the sequencer. Then the host reads back the output FIFO.

Think about the new `OpType` enum:

```
  OpType:
    load_weights = 0x01    // bulk-load weight data into filter BSRAM
    load_acts    = 0x02    // bulk-load activation data into act BSRAM
    load_params  = 0x03    // load requant params into param BSRAM
    execute      = 0x04    // start autonomous compute
    read_output  = 0x05    // drain output FIFO
```

Compare this to CUDA's kernel launch sequence: `cuMemcpyHtoD()` (load data), `cuLaunchKernel()` (execute), `cuMemcpyDtoH()` (read results). The structure is identical.

---

## 4.11  Checkpoint

After completing this unit, you should have:

- [ ] Config registers that the CPU writes before compute (the compute descriptor)
- [ ] BSRAM filter stores with cyclic read and `first`/`last` signal generation
- [ ] BSRAM activation stores with sequential read
- [ ] A sequencer FSM: IDLE -> RUNNING -> DONE
- [ ] An output FIFO that the CPU drains after DONE
- [ ] Correct INT8 output for a known matmul (compare against NumPy reference)
- [ ] The CPU does NO work during the compute phase

**The test:** CPU writes config + START, waits for DONE, drains FIFO. Correct INT8 output for a known input. Zero CPU involvement during compute.

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

### Recommended order

1. **Command queue** — biggest bang for complexity. Eliminates per-tile CPU round trips.
2. **Double-buffer** — nearly free given existing 2 filter BSRAMs. Layer on top of command queue.
3. **2D tiling** — needed when MobileNet layers don't fit. Orthogonal to the above.
4. **Async load / layer fusion** — powerful but high complexity. Pursue if you want to push utilization toward theoretical limits.

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
