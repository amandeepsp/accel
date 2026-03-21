# Unit 7: Spatial Parallelism — Systolic Arrays and Tensor Cores

> **Course:** [00-architecture](00-architecture.md) > [01-compute](01-compute.md) > [02-vertical-slice](02-vertical-slice.md) > [03-fusion](03-fusion.md) > [04-engine](04-engine.md) > [05-compiler](05-compiler.md) > [06-model](06-model.md) > **[07-systolic](07-systolic.md)**
> **Reference:** [Prior Art & Architecture Decisions](appendix-prior-art.md)

---

An NVIDIA Tensor Core does a 4x4x4 matrix multiply per cycle. Google's TPU v1 has a 128x128 systolic array. You're building a 4x4 -- same concept, 1000x smaller.

The previous units gave you a working accelerator with a single 4-wide SIMD MAC. This unit replaces that single pipeline with a grid of processing elements that compute in parallel, turning one matmul cycle into sixteen.

---

## 7.1  When to Scale

Before building anything, measure what you have. From your Unit 4 autonomous engine:

```
  Single 4-wide MAC, 27 MHz:
    4 MACs/cycle x 27 MHz = 108 MOPS peak

  MobileNet v2 0.25, 96x96 input:
    ~5.6 million MACs total
    At 108 MOPS: ~52 ms per inference

  Is 52 ms acceptable?
    Real-time video (30 fps): need < 33 ms. Close, but not there.
    Still-image classification: plenty fast. No array needed.
```

A systolic array adds ~800-2000 lines of HDL complexity. Make sure the payoff justifies the effort.

> **MLSys Connection:** This is the same calculus GPU architects make. NVIDIA doesn't add Tensor Cores to every GPU -- only the ones targeting ML workloads (datacenter, workstation). Consumer GPUs got Tensor Cores only after the workload justified the die area. Specialized compute is expensive in silicon area and design effort. The question is always: does the throughput gain justify the cost?

---

## 7.2  What Is a Systolic Array?

A systolic array is a grid of simple processing elements (PEs) connected in a regular pattern. Data flows through the grid like a pulse through tissue (hence "systolic," from the Greek for contraction).

### A Single PE

Each PE has four things: a weight register, a multiplier, an adder, and a passthrough for activations.

```
                  +-----------------------------------+
   weight_load -->|   +-------+                       |
                  |   |weight |  (loaded once)        |
                  |   |  reg  |                       |
                  |   +---+---+                       |
                  |       |                           |
   act_in ------->|---+---+-------------------------------> act_out
                  |   |   v                           |
                  |   | +-----+                       |
                  |   +>|  x  |  INT8 x INT8          |
                  |     +--+--+                       |
                  |        v                          |
   psum_in ------>|--> +-----+                        |
                  |    |  +  |  16 + 32 -> 32         |
                  |    +--+--+                        |
                  |       v                           |
                  |   psum_out                        |
                  +-----------------------------------+
```

The activation passes through horizontally (so the next PE in the row sees it). The partial sum flows vertically (accumulating down the column). The weight stays put.

### The 4x4 Array

```
          +-----+  +-----+  +-----+  +-----+
  act --> |PE0,0|--> PE0,1|--> PE0,2|--> PE0,3|-->
          +--+--+  +--+--+  +--+--+  +--+--+
  act --> |PE1,0|--> PE1,1|--> PE1,2|--> PE1,3|
          +--+--+  +--+--+  +--+--+  +--+--+
  act --> |PE2,0|--> PE2,1|--> PE2,2|--> PE2,3|
          +--+--+  +--+--+  +--+--+  +--+--+
  act --> |PE3,0|--> PE3,1|--> PE3,2|--> PE3,3|
          +--+--+  +--+--+  +--+--+  +--+--+
             v        v        v        v
         (partial sums -- 4 output columns)
```

- **16 PEs x 1 DSP each = 16 DSPs** (33% of your 48-DSP budget)
- Plus 4 DSPs for requantization = 20 DSPs (42%)
- Remaining: 28 DSPs free

> **MLSys Connection:** Google's TPU v1 is a 256x256 systolic array (65,536 PEs). NVIDIA's Tensor Cores are 4x4x4 units replicated across thousands of CUDA cores. Apple's Neural Engine uses a similar grid structure. The scale differs by 1000x, but the PE design is nearly identical: multiply, accumulate, pass data to neighbor. Understanding a 4x4 array means you understand the TPU -- just add zeros to the dimensions.

---

## 7.3  The Key Question: Weight-Stationary or Output-Stationary?

This is the most important architectural decision for a systolic array. It determines what stays pinned in the PEs and what flows through.

### Weight-Stationary (WS)

Weights are loaded into PE registers once per tile. Activations stream left-to-right. Partial sums flow top-to-bottom.

```
  Pinned:   weights (in PE registers)
  Flows:    activations (horizontal), partial sums (vertical)
  Reload:   weights must be reloaded for each output tile

  Advantage:  Simple control. Weights loaded once, reused across many activations.
  Disadvantage: Must tile the K dimension. CPU must manage partial sums across tiles.
```

### Output-Stationary (OS)

Accumulators stay pinned in the PEs. Both weights and activations flow through.

```
  Pinned:   accumulators (partial sums stay in PE)
  Flows:    activations (horizontal), weights (vertical)
  Reload:   nothing -- accumulator resets via first/last signals

  Advantage:  No partial-sum management. K dimension handled in one pass.
              first/last signals eliminate explicit reset/read cycles.
  Disadvantage: Weights must stream continuously. Needs BSRAM filter stores.
```

Google's `hps_accel` chose output-stationary. Why? Because the accumulator is the most precious value -- it is 32 bits of precision being built up over many cycles. Moving it between PEs or reading it out and writing it back risks precision loss and adds control complexity. Pinning the accumulator and streaming everything else is cleaner.

> **MLSys Connection:** The dataflow taxonomy (weight-stationary, output-stationary, row-stationary) was formalized by the MIT Eyeriss project. Every ML accelerator makes this choice. TPU v1 is weight-stationary. Eyeriss v1 is row-stationary. NVDLA is output-stationary for certain configurations. The "right" answer depends on the memory hierarchy, bandwidth constraints, and target workload. There is no universally optimal dataflow.

### Exercise: Choose Your Dataflow

Pick weight-stationary or output-stationary for your 4x4 array. Justify your choice based on:
1. Your BSRAM layout (how many filter stores do you have?)
2. Your sequencer design (does it already support first/last signals?)
3. The K dimensions of MobileNet v2 0.25 layers (are they small enough for single-pass OS?)

<details><summary>Hint 1</summary>

If you built the autonomous sequencer from Unit 4 with first/last accumulator control, output-stationary is a natural fit -- you already have the control signals. If your sequencer manages explicit K-dimension tiles, weight-stationary is simpler to integrate.

</details>

<details><summary>Hint 2</summary>

For MobileNet v2 0.25, the largest K dimension (input depth) in a pointwise conv is 32. With a 4-wide SIMD MAC, that is 8 cycles of accumulation -- easily handled in a single pass with output-stationary. You do not need K-dimension tiling for this model.

</details>

<details><summary>Solution</summary>See solutions/07-systolic/dataflow_analysis.md</details>

---

## 7.4  Building the Array

Build incrementally. Do NOT skip to the full array. Each step catches different bugs.

### Step 1: Single PE

Implement one processing element as an Amaranth module.

**Interface:**
```
  Inputs:  act_in (signed 8), psum_in (signed 32), weight_load (signed 8), load_en (1)
  Outputs: act_out (signed 8), psum_out (signed 32)
```

**Behavior:**
- When `load_en` is high, latch `weight_load` into the weight register
- Every cycle: `psum_out = psum_in + (act_in * weight_reg)`
- Every cycle: `act_out = act_in` (passthrough, 1-cycle delay)

**Test:** Feed known `act_in`, `psum_in`, and a loaded weight. Verify `psum_out = psum_in + act_in * weight`.

<details><summary>Hint 1</summary>

The multiply should infer a single DSP block (MULT9X9 on Gowin). If synthesis puts it in LUTs, check that your operands are `signed(8)` -- the synthesizer needs to see the right width to infer DSP.

</details>

<details><summary>Hint 2</summary>

The activation passthrough must be registered (1-cycle delay), not combinational. This is what creates the "wave" of data flowing through the array. If the passthrough is combinational, timing will cascade across all columns.

</details>

<details><summary>Solution</summary>See solutions/07-systolic/pe.py</details>

### Step 2: 1D Row (4 PEs)

Wire 4 PEs in a row. `act_out` of PE[i] connects to `act_in` of PE[i+1]. Each PE has an independent `psum_in` (from the row above, or 0 for the top row).

**Test:** Load 4 different weights into the 4 PEs. Stream a sequence of activations into PE[0]. Verify that each PE computes the correct partial sum. This is a dot product spread across space and time.

<details><summary>Hint 1</summary>

Because of the 1-cycle activation delay per PE, the activation reaches PE[0] at cycle 0, PE[1] at cycle 1, PE[2] at cycle 2, PE[3] at cycle 3. The partial sums at each PE are offset in time accordingly. You need to account for this skew when collecting results.

</details>

<details><summary>Solution</summary>See solutions/07-systolic/row.py</details>

### Step 3: 2D Array (4x4)

Stack 4 rows. `psum_out` of PE[r,c] connects to `psum_in` of PE[r+1,c].

**Critical: activation skew.** To get correct matrix multiplication, the activations entering each row must be staggered:

```
  Cycle:    0     1     2     3     4     5     6
  Row 0:   a0,0  a0,1  a0,2  a0,3   -     -     -
  Row 1:    -    a1,0  a1,1  a1,2  a1,3   -     -
  Row 2:    -     -    a2,0  a2,1  a2,2  a2,3   -
  Row 3:    -     -     -    a3,0  a3,1  a3,2  a3,3
```

Row `i` starts `i` cycles after row 0. This skew ensures that each PE receives the correct activation at the correct time to compute `C[i,j] = sum_k(A[i,k] * B[k,j])`.

Total cycles from first input to last output: **K + (rows - 1) + (cols - 1) = K + 6** for a 4x4 array.

**Test:** Compute `C = A @ B` for random 4x4 matrices. Compare against `numpy.matmul`. Try several random cases.

<details><summary>Hint 1</summary>

The skew can be implemented with shift registers (FIFOs) at each row input. Row 0 gets a 0-deep FIFO (direct), row 1 gets a 1-deep FIFO, row 2 gets 2-deep, row 3 gets 3-deep. Each FIFO delays the activation stream by the right amount.

</details>

<details><summary>Hint 2</summary>

Off-by-one timing in the skew is the most common systolic array bug. If your 4x4 test passes for `K=1` but fails for `K>1`, the skew is wrong. Draw the timing diagram for `K=4` on paper, marking exactly which cycle each PE sees each activation.

</details>

<details><summary>Solution</summary>See solutions/07-systolic/array.py</details>

### Step 4: Integration

Wire the 4x4 array to your BSRAM stores and autonomous sequencer from Unit 4.

- Filter BSRAM feeds weights to each column
- Activation BSRAM feeds activations to each row (with skew)
- Partial sum outputs at the bottom connect to your post-processing pipeline
- The sequencer drives first/last signals and address counters

<details><summary>Hint 1</summary>

The sequencer needs to know the array dimensions (4x4) to generate the correct address patterns. For the activation skew, the sequencer can either: (a) read from 4 different BSRAM addresses offset by the skew amount, or (b) read sequentially and let hardware skew FIFOs handle the delay.

</details>

<details><summary>Solution</summary>See solutions/07-systolic/integration.py</details>

---

## 7.5  Data Tiling for Layers Larger Than 4x4

For layers with more than 4 output channels or more than 4 input positions per batch, you must tile:

```
  C[M x N] = A[M x K] x B[K x N]

  Tile B into ceil(N/4) column-strips (weight tiles)
  Tile A into ceil(M/4) row-strips (activation tiles)

  Total tile operations: ceil(M/4) x ceil(N/4)
  Each tile: K + 6 cycles of array compute + weight reload
```

### Exercise: Tile the First Pointwise Conv

MobileNet v2 0.25's first pointwise conv:
- 1x1 conv, 16 input channels, 8 output channels, 48x48 spatial
- In matmul terms: M = 48x48 = 2304, K = 16, N = 8

Calculate:
1. How many tiles? (`ceil(2304/4) x ceil(8/4)`)
2. Cycles per tile? (`K + 6 = 22`)
3. Total compute cycles?
4. Time at 27 MHz?

<details><summary>Hint 1</summary>

Tiles: `ceil(2304/4) x ceil(8/4) = 576 x 2 = 1152` tile operations. Each tile: `16 + 6 = 22` cycles. Total: `1152 x 22 = 25,344` cycles. At 27 MHz: `25,344 / 27e6 = 0.94 ms`.

</details>

<details><summary>Hint 2</summary>

Compare against the single-MAC version from Unit 4: `2304 x 8 x 16 / 4 = 73,728` cycles = 2.73 ms at 27 MHz. The 4x4 array is ~2.9x faster. Not the full 4x because of tiling overhead (weight reloading, skew fill/drain cycles).

</details>

<details><summary>Solution</summary>See solutions/07-systolic/tiling_calc.py</details>

### Exercise: Modify the Sequencer

Update your autonomous sequencer to iterate over tiles:

```
  for each weight tile (column strip):
      load weight tile into filter BSRAM
      for each activation tile (row strip):
          load activation tile into act BSRAM
          execute array for K + 6 cycles
          drain partial sums to post-processing
```

<details><summary>Hint 1</summary>

The outer loop (weight tiles) reloads filter BSRAM. The inner loop (activation tiles) reloads activation BSRAM. Weights are reloaded less often because they are shared across all spatial positions. This is the standard "loop ordering for data reuse" optimization.

</details>

<details><summary>Solution</summary>See solutions/07-systolic/tiled_sequencer.py</details>

> **MLSys Connection:** Tiling for a systolic array is the same problem as tiling for GPU shared memory. In CUDA, you load tiles of A and B into shared memory, compute a partial result, then load the next tiles. The tile size is determined by shared memory capacity (48 KiB on Ampere). In XLA, the `dot_dimension_numbers` and `lhs_contracting_dimensions` in the HLO tell the compiler how to tile. In TVM, `schedule.tile()` does this explicitly. Your sequencer FSM is a hardware implementation of what CUDA's threadblock-level tiling does in software.

---

## 7.6  Performance Scaling

### Compute Scaling

| Configuration | MACs/cycle | Peak MOPS @ 27 MHz |
|---|---|---|
| Single MAC4 (Unit 1) | 4 | 108 |
| 4x4 systolic array | 16 | 432 |
| **Speedup** | **4x** | **4x** |

### Clock Scaling

The PLL on the Tang Nano 20K can multiply the 27 MHz crystal:

```
  27 MHz  ->  432 MOPS peak  (baseline with 4x4 array)
  54 MHz  ->  864 MOPS peak  (2x -- usually closes timing)
  81 MHz  -> 1296 MOPS peak  (3x -- may need pipeline registers)
 108 MHz  -> 1728 MOPS peak  (4x -- likely needs work)
```

### Exercise: Find the Clock Ceiling

Change `sys_clk_freq` in the SoC configuration. Rebuild. Check the synthesis report:
- Does timing close?
- What is the critical path? Is it in the array or in VexRiscv?
- At what frequency does it fail?

If the critical path is in the systolic array, add pipeline registers to PE outputs. This adds 1 cycle of latency per PE but shortens the combinational chain.

<details><summary>Hint 1</summary>

The Gowin synthesis report shows the critical path in the "Timing Analysis" section. Look for the longest combinational path. On a 4x4 array without pipeline registers, the critical path is likely the partial-sum adder chain across 4 rows.

</details>

<details><summary>Solution</summary>See solutions/07-systolic/clock_scaling.md</details>

### The Memory Wall

Your 4x4 array can consume data much faster than PSRAM can deliver:

```
  Array demand:  16 MACs/cycle -> needs 16 bytes activations + 16 bytes weights
                 = 32 bytes/cycle x 27 MHz = 864 MB/s

  BSRAM supply:  32 bits/cycle x 27 MHz = 108 MB/s per port
                 With 4 activation banks: 432 MB/s
                 With 2 filter stores:    216 MB/s

  BSRAM total:   648 MB/s (enough at 27 MHz, tight at higher clocks)

  PSRAM supply:  ~40-100 MB/s (burst)
                 864 MB/s needed vs 100 MB/s available = MEMORY BOUND
```

BSRAM is fast enough to feed the array at 27 MHz. But if you need to reload from PSRAM between tiles, PSRAM becomes the bottleneck.

> **MLSys Connection:** This is the memory wall that dominates all ML hardware design. NVIDIA's A100 has 2 TB/s of HBM bandwidth, and it is *still* memory-bound on many transformer workloads. Google's TPU v4 has 1.2 TB/s. The ratio of compute (FLOPS) to memory bandwidth (bytes/s) is called the "arithmetic intensity" threshold -- ops that fall below it are memory-bound. Your PSRAM bottleneck at 100 MB/s is the same problem, just at a different scale. The solutions are also the same: tiling for data reuse, double-buffering, compression.

### What Helps?

- **Double-buffering:** Load the next tile into BSRAM from PSRAM while computing the current tile. Overlaps transfer and compute.
- **Weight compression:** 4-bit weights halve the bandwidth needed for weight loading.
- **Tiling for reuse:** Maximize data reuse within BSRAM before going back to PSRAM. Process all spatial positions for one weight tile before loading the next.
- **DMA engine:** Dedicated hardware for PSRAM-to-BSRAM transfers, freeing the CPU.

---

## 7.7  GPU Parallel: How This Maps to Real Hardware

Your 4x4 systolic array computes `C[4x4] = A[4xK] x B[Kx4]` in `K + 6` cycles. Here is how that compares to production hardware:

| | Your Array | NVIDIA Tensor Core | Google TPU v1 |
|---|---|---|---|
| **Array size** | 4x4 | 4x4x4 | 256x256 |
| **MACs/cycle** | 16 | 64 (per TC) | 65,536 |
| **Dataflow** | WS or OS | Matrix multiply unit | Weight-stationary |
| **Data type** | INT8 | INT8/FP16/TF32/FP8 | INT8 |
| **Clock** | 27-108 MHz | ~1.4 GHz | ~700 MHz |
| **On-chip SRAM** | ~18 KiB (BSRAM) | 256 KiB (shared mem per SM) | 24 MiB (unified buffer) |
| **Peak TOPS** | 0.0004-0.002 | ~330 (A100) | ~92 (v1) |

The NVIDIA Tensor Core does a 4x4x4 in one cycle -- it computes the entire K=4 reduction in a single shot. Your 4x4 array takes K cycles for the reduction. Same spatial parallelism, different temporal depth.

The TPU v1's 256x256 array is 4096x larger than yours. But each PE is functionally identical: multiply, accumulate, pass data to neighbor.

> **MLSys Connection:** Understanding systolic arrays at this level is directly applicable to reasoning about TPU performance. When Google reports "275 TOPS" for TPU v5e, that is `256 * 256 * 2 ops/MAC * clock_freq`. When NVIDIA reports Tensor Core throughput, it is `4 * 4 * 4 * 2 * num_TCs * clock_freq`. The architecture scales linearly. The challenges (memory bandwidth, tiling, utilization) scale too.

---

## 7.8  Checkpoint

Run the matmul benchmark from Unit 4 on your 4x4 systolic array. Compare against the single-MAC baseline.

- [ ] Single PE passes unit tests (multiply-accumulate + passthrough)
- [ ] 1D row computes correct dot products
- [ ] 4x4 array matches `numpy.matmul` on random test cases
- [ ] Activation skew timing is correct (draw the diagram, verify with simulation)
- [ ] Array is integrated with BSRAM stores and sequencer
- [ ] Tiling works for layers larger than 4x4
- [ ] **4x throughput improvement** on the matmul benchmark from Unit 4
- [ ] I can explain weight-stationary vs output-stationary tradeoffs
- [ ] I understand the memory bandwidth bottleneck and what helps

---

**Previous:** [Unit 6 -- End-to-End: Run a Real Model](06-model.md)
**Reference:** [Prior Art & Architecture Decisions](appendix-prior-art.md)

## Side Quests

- **Output-stationary dataflow.** Implement output-stationary (accumulator stays in PE, both weights and activations flow) and compare against weight-stationary. Output-stationary eliminates K-dimension tiling — the `first`/`last` pattern from the appendix. Measure: control complexity, BSRAM usage, and utilization.
- **Sparse acceleration.** If a weight is zero, skip the multiply. Add a zero-detection circuit that advances the filter store address without consuming a compute cycle. Pruned MobileNet models have 30–70% zero weights — this can nearly double effective throughput. Read [Han et al., "Deep Compression"](https://arxiv.org/abs/1510.00149) for context.
- **Roofline analysis.** Build a roofline model for your specific hardware: 27 MHz, 16 MACs/cycle peak, 108 MB/s BSRAM bandwidth. Plot where each MobileNet layer falls. Identify which layers are compute-bound vs. memory-bound. The roofline tells you exactly when scaling the array helps vs. when you need more bandwidth.
- **Clock pushing.** Use the PLL to push clock speed: 27 -> 54 -> 81 -> 108 MHz. Add pipeline registers to the PE critical path if timing fails. Document the frequency ceiling and what limits it (array? VexRiscv? routing?). This is the cheapest 2–4x speedup available.
- **2x2 first.** Build a 2x2 array, get it fully passing all tests, then scale to 4x4. The 2x2 array catches every timing and control bug with 4x fewer signals to debug. The jump from 2x2 to 4x4 should be parameterized — change one constant, resynthesize.

---

## Suggested Readings

| Topic | Source |
|---|---|
| Google TPU v1 (systolic arrays) | [arxiv.org/abs/1704.04760](https://arxiv.org/abs/1704.04760) -- Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit" |
| Eyeriss (dataflow taxonomy) | [eyeriss.mit.edu](https://eyeriss.mit.edu/) -- Sze et al., row-stationary dataflow and energy analysis |
| Eyeriss v2 (flexible dataflow) | [arxiv.org/abs/1807.07928](https://arxiv.org/abs/1807.07928) -- Chen et al., hierarchical mesh for sparse acceleration |
| NVIDIA Tensor Core architecture | [arxiv.org/abs/1811.01143](https://arxiv.org/abs/1811.01143) -- Markidis et al., "NVIDIA Tensor Core Programmability, Performance & Precision" |
| Systolic array tutorial | [cs.utexas.edu/~pingali/](https://www.cs.utexas.edu/~pingali/CS378/2008sp/papers/Kung.pdf) -- H.T. Kung, "Why Systolic Architectures?" (the original 1982 paper) |
| CFU-Playground hps_accel | [github.com/google/CFU-Playground](https://github.com/google/CFU-Playground) -- `proj/hps_accel/gateware/gen2/` for the 4x2 systolic implementation |
| Gowin pDSP user guide | UG289E -- MULT9X9, MULT18X18, MULTALU primitive configurations |
| Data reuse and tiling | [arxiv.org/abs/1602.04183](https://arxiv.org/abs/1602.04183) -- Chen et al., "Eyeriss: An Energy-Efficient Reconfigurable Accelerator" (Section IV on dataflow) |
| Sparse accelerators | [arxiv.org/abs/1510.00149](https://arxiv.org/abs/1510.00149) -- Han et al., "Deep Compression: Compressing DNNs with Pruning, Trained Quantization and Huffman Coding." Foundational work on model sparsity |
| Roofline model | [doi.org/10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785) -- Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model" (2009). The framework for understanding whether you're compute-bound or memory-bound |
| FPGA systolic array case study | [arxiv.org/abs/1807.06434](https://arxiv.org/abs/1807.06434) -- Abdelfattah et al., "DLA: Compiler and FPGA Overlay for Neural Network Inference Acceleration" (2018). A systolic array on Intel FPGAs — compare their design choices to yours |
