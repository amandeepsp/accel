# Unit 4: The Execution Engine — Control Planes, Local Memory, and Scheduling

> **Series:** [00-architecture](00-architecture.md) → [01-compute](01-compute.md) → [02-datapath](02-datapath.md) → [03-fusion](03-fusion.md) → **[04-engine](04-engine.md)** → [05-compiler](05-compiler.md) → [06-model](06-model.md) → [07-systolic](07-systolic.md) → [08-feeding](08-feeding.md) → [09-modern](09-modern.md) → [10-redesign](10-redesign.md)

Unit 3 taught why keeping intermediate values on-chip matters. Unit 4 asks the bigger accelerator question: when does a datapath become an execution engine?

Not when it gets faster. When it starts to own local memory, control state, and the loop nest.

This is also the course's explicit control-plane unit: custom instructions, register writes, descriptors, firmware loops, and hardware FSMs all belong to the same design question.

> **Status:**
> - `Implemented in repo:` instruction-level control plane, `mac4`, accumulator read/reset, `input_offset`, `srdhm`, `rdbpot`
> - `Design exercise:` local operand stores, `mac4_next`, output FIFO, compute descriptor
> - `Stretch:` hardware sequencer / autonomous engine

---

## 4.1  A Datapath Is Not Yet an Engine

By the end of Unit 3, you have arithmetic blocks and a control path. That is enough to build a useful datapath. It is not yet enough to call the hardware an execution engine.

| Responsibility | Datapath | Execution Engine |
|---|---|---|
| Operand delivery | CPU streams operands each step | Hardware reads from local memory |
| Loop control | CPU owns every iteration | Device owns some or all of the loop nest |
| Intermediate state | Register / accumulator only | Local stores + control state + output buffering |
| Launch contract | One op at a time | One descriptor or command launches a region of work |
| Host role | Drive the hot path | Configure, launch, synchronize, drain |

That is the core lesson of this unit: an accelerator is not just arithmetic. It is arithmetic plus memory plus scheduling.

> **MLSys Connection:** Real accelerators are shaped more by data movement and control ownership than by any single multiplier. A TPU matrix unit, a GPU SM, and an NVDLA convolution core all look different internally, but each one couples compute with local storage and a machine that decides what happens next.

---

## 4.2  Three Control Models and Control Planes

This course is easiest to understand if you treat execution control as a ladder.

| Model | Who owns the inner loop? | What you learn | Why real hardware uses or avoids it |
|---|---|---|---|
| CPU as data pump | CPU | Baseline control cost and launch overhead | Useful for bring-up, terrible for throughput |
| CPU as sequencer | CPU firmware | How local memory and fixed-function compute interact | Common as a teaching step and bring-up strategy |
| Hardware sequencer | Dedicated hardware | What "autonomous execution" really means | How high-throughput accelerators actually ship |

The mistake is to think only the last one is worth studying. The middle step is extremely educational because it isolates the control contract from the scheduler implementation.

Another useful view is to ask what software is *actually* talking to:

| Control plane | Software sees | Typical strength | Typical weakness |
|---|---|---|---|
| Custom instructions | one operation at a time | very easy bring-up | terrible batching |
| MMIO / config registers | a small fixed-function engine | simple host contract | limited expressiveness |
| Descriptors | work packets and queues | good amortization | compiler/runtime must know more |
| Microcode / tiny instruction streams | a semi-programmable engine | flexibility | more control complexity |

Real accelerators pick among these, or combine them. The right choice depends on workload diversity, latency targets, and how much intelligence you want in hardware versus software.

### CPU as data pump

This is where the repo sits today. The host or firmware explicitly issues `mac4`, `srdhm`, and `rdbpot` step by step.

### CPU as sequencer

The CPU stops feeding raw operands and starts issuing higher-level progress commands like `mac4_next`, while local stores feed the datapath. This is not the final form of a serious accelerator, but it is a great place to learn.

### Hardware sequencer

Now the host launches work and leaves. The device walks `spatial x N x K`, manages local addresses, and drains results into a buffer or FIFO.

> **Teaching point:** For this course, CPU-as-sequencer is not a compromise to hide. It is a deliberate intermediate architecture that teaches the boundary between control software and fixed-function hardware.

---

## 4.3  What Real Accelerators Actually Do

The right way to teach analogies is to be honest about where they hold and where they break.

| Course idea | Do real accelerators do this? | Why or why not? | Honest framing for this course |
|---|---|---|---|
| Host/device split | Yes | General-purpose software and specialized compute want different designs | This analogy is real and foundational |
| Custom instruction interface | Sometimes | Great for tightly-coupled edge accelerators, uncommon for large GPU-style launch paths | Treat CFU as the simplest possible control plane |
| Software-managed local memory | Yes | Scratchpads/shared memory give predictable reuse and bandwidth | This is one of the most transferable ideas in the course |
| Fused epilogue (`bias -> scale -> shift -> clamp`) | Yes | These ops are bandwidth-dominated and naturally sit on the output path | Epilogue fusion is a real and high-value special case |
| CPU in the hot inner loop | Usually no | It burns energy and throughput on control overhead | Useful for learning and bring-up, not where production hardware stops |
| Descriptor-driven fixed-function engine | Yes, often | Descriptors are compact and efficient when the dataflow is known | Real systems often mix descriptors with programmable control cores |
| Request/response over UART | No | Real accelerators use DMA, queues, interrupts, and many in-flight commands | Here the slowness is a teaching tool, not a model of production transport |

If you keep that table in mind, the whole course gets sharper: the goal is not to cosplay an A100. The goal is to understand why these mechanisms exist.

---

## 4.4  What Exists in This Repo Today

Before designing the next step, anchor on the current implementation.

- `hardware/top.py` wires together `SimdMac4`, register read/write instructions, `SRDHMInstruction`, and `RoundingDividebyPOTInstruction`
- `hardware/mac.py` already has a 4-lane INT8 MAC with accumulator state and configurable `input_offset`
- `hardware/rw.py` already exposes `input_offset` writes and accumulator reset / readback
- `shared/cfu.zig` exposes `mac4`, `set_input_offset`, `reset_accumulator`, `read_accumulator`, `srdhm`, and `rdbpot`
- `firmware/dispatch.zig` already handles instruction-at-a-time requests over the current protocol
- `driver/driver.zig` and `shared/protocol.zig` already define the host/device contract for those operations

What is not there yet is exactly what makes Unit 4 interesting:

- local BSRAM-backed operand stores
- a `mac4_next` path that reads from those stores
- a compute descriptor or equivalent launch contract
- an output FIFO or buffered drain path
- a hardware sequencer that can run without CPU loop overhead

---

## 4.5  Local Memory Is the Real Architectural Step

Real accelerators live or die by their on-chip working set. In GPUs this is shared memory, registers, and caches. In your FPGA design it is BSRAM.

That makes local memory the first real architectural step from datapath to engine.

One reasonable design target for the Tang Nano 20K is:

| Purpose | BSRAM blocks | Why it exists |
|---|---|---|
| Activation scratchpad | 4 | Reuse activations across output channels |
| Filter scratchpad | 2 | Keep weights local during a tile |
| Requant / bias params | 2 | Per-channel post-processing constants |
| Output FIFO / packed buffer | 1 | Decouple compute from host drain |
| Spare | 7+ | Ping-pong buffering, wider tiles, experiments |

This exact budget is not sacred. The important lesson is that local SRAM capacity determines tile shape, reuse strategy, and control complexity.

### Why real accelerators do this

- Local SRAM is far cheaper to access than off-chip memory.
- Reuse inside the tile is what makes arithmetic units worth building.
- Scratchpads are predictable. Caches are convenient, but accelerators often prefer explicit control.

### Where this course simplifies reality

The course often shows the CPU writing these stores directly. That is fine for learning, but real systems often use DMA engines, copy queues, or compiler-scheduled prefetches instead.

---

## 4.6  The Minimal Execution Contract

Once local memory exists, the engine needs a contract. A compiler or runtime has to know what must be configured before execution starts.

| Piece | Minimal first version | Why it matters |
|---|---|---|
| Activation store | Packed INT8 words in BSRAM | Reuse one activation tile across many MACs |
| Filter store | Packed INT8 words in BSRAM | Feed the MAC without CPU operand streaming |
| Param store | Bias / multiplier / shift per channel | Support real quantized post-processing |
| Descriptor | `K`, `N`, `spatial`, offsets, clamp bounds | Defines the iteration space and output transform |
| Output path | FIFO or packed result buffer | Decouples compute from drain |
| Control model | Firmware sequencer first, FSM later | Separates the execution contract from the final scheduler |

This is where Unit 4 starts to connect directly to Unit 5: a compiler backend is really just a way of producing data and metadata that satisfy this contract.

### Exercise 4a: Design the compute descriptor

Decide the smallest set of launch parameters that lets the device compute a full 1x1 INT8 conv tile without CPU intervention inside the inner `K` loop.

Questions to answer:

- What must the device know to stop the `K` loop?
- What must it know to advance output channels?
- Which quantization values are per-launch and which are per-channel?
- Should `spatial` be explicit, or derived from a higher-level tile description?

If you want a concrete starting point, a minimal first descriptor usually includes:

- `K`
- `N`
- `spatial`
- `input_offset`
- `output_offset`
- `activation_min`
- `activation_max`

Per-channel bias, multiplier, and shift can live in a separate parameter store indexed by output channel.

---

## 4.7  Exercise 4b: BSRAM Stores and the Prefetch Pattern

Synchronous SRAM matters here. You present an address on one edge, and the data shows up on the next.

That means your engine needs a prefetch pattern:

```text
Cycle -1: addr <- 0
Cycle  0: data[0] available, MAC consumes it, addr <- 1
Cycle  1: data[1] available, MAC consumes it, addr <- 2
Cycle  2: data[2] sits ready while loop overhead happens
Cycle  3: same data[2] still available
Cycle  4: MAC consumes data[2], addr <- 3
```

That one timing fact is enough to teach several accelerator truths:

- local memory is only useful if the access pattern is designed with the datapath
- address generation is part of the architecture
- simple fixed-function dataflows can hide latency surprisingly well

For a first version, give each store:

- one CPU-visible write path
- one hardware read pointer
- an explicit reset / prime operation
- optional cyclic wrap for the filter store

One implementation sketch already lives in `docs/course/solutions/04-engine/bsram_stores.md`.

> **MLSys Connection:** This is the same reason GPU and TPU papers spend so much time on tile iterators, bank layout, and prefetch. Feeding the array is architecture, not plumbing.

---

## 4.8  Exercise 4c: CPU as Sequencer

Once the stores exist, the most educational next step is usually not a giant FSM. It is a higher-level instruction that tells the datapath to pull from local memory instead of CPU registers.

Call it `mac4_next`.

Conceptually it does five things:

1. read the next packed activation word from local memory
2. read the next packed filter word from local memory
3. feed both into the existing `SimdMac4`
4. advance one or both read pointers
5. optionally generate `first` / `last` style control for accumulator reset and output emission

One possible ABI extension looks like this. The current repo already reads and writes small control registers through the existing `ReadRegs` / `WriteRegs` path; this table shows how that interface could grow into an engine-oriented contract:

```text
funct3  funct7  Meaning
0       0       MAC4 from CPU operands           (implemented now)
0       1       MAC4_NEXT from local stores      (design target)
1       0       READ reg selected by `in0`       (implemented now)
2       0       WRITE reg selected by `in0`      (implemented now)
2       1       RESET accumulator                (optional cleaner variant)
2       2       WRITE k_count                    (design target)
2       3       WRITE filter store               (design target)
2       4       WRITE activation store           (design target)
```

The inner loop becomes firmware, not host traffic:

```zig
fn computeTile(k_count: u16, n_channels: u16, spatial: u16) void {
    for (0..spatial) |_| {
        for (0..n_channels) |n| {
            _ = n;
            cfu.reset_accumulator();
            cfu.reset_act_rptr();

            var k: u16 = 0;
            while (k < k_count) : (k += 4) {
                _ = cfu.mac4_next();
            }

            const acc = cfu.read_accumulator();
            // Requantization path can still be firmware-driven at first.
            // Later it can move fully into hardware.
            _ = acc;
        }
    }
}
```

Why this is a good learning step:

- the launch contract gets clearer
- local memory now matters
- the CPU is no longer a raw operand pump
- the eventual FSM is easier to design because the execution contract already exists

Why real accelerators do not stop here:

- a general-purpose CPU in the hot loop still wastes cycles and energy on control
- hardware scheduling gives more consistent utilization
- firmware loops are great for debug, not ideal for production throughput

One possible implementation sketch already lives in `docs/course/solutions/04-engine/sequencer_fsm.md`.

---

## 4.9  Exercise 4d: Add an Output FIFO

An engine without a buffered output path is still partly coupled to the reader.

That is why output FIFOs show up everywhere:

- between pipelines inside GPUs
- at the edges of DMA engines
- in streaming FPGA datapaths
- in accelerator blocks that produce results faster than software can drain them

For this course, the FIFO matters even more because the link is intentionally slow.

Minimal requirements:

- write pointer on the hardware side
- read pointer on the software side
- empty / full detection
- packing policy for INT8 outputs into 32-bit words
- a host-visible read operation

This section is worth doing even if the end-to-end system is still slow. The lesson is not raw speed. The lesson is decoupling producer and consumer timing.

---

## 4.10  Exercise 4e: Replace the Firmware Loop With a Hardware Sequencer

This is the point where the design becomes an actual autonomous engine.

At minimum the sequencer owns:

- a `k_counter`
- an `n_counter`
- a `spatial_counter`
- address generation for the local stores
- output emission timing
- `busy` / `done` state

A simple state machine is enough:

```text
IDLE -> LOAD/PRIME -> RUN -> DRAIN -> DONE
```

Inside `RUN`, the hardware walks the same loop nest that firmware used to own.

### Why real accelerators do this

- better steady-state utilization
- lower energy per useful operation
- less control overhead
- easier overlap with prefetch, DMA, or output drain

### Where the course still simplifies reality

Real accelerators often use something between a hard FSM and a general CPU:

- microcoded controllers
- small scalar cores next to tensor datapaths
- command processors plus specialized engines

The FSM is still the right teaching device because it makes the scheduling responsibilities explicit.

### Compare the control models honestly

| Property | CPU as data pump | CPU as sequencer | Hardware sequencer |
|---|---|---|---|
| Simplest bring-up | Yes | Moderate | No |
| Easiest to debug | Yes | Yes | Harder |
| Best for learning contracts | No | Yes | Yes |
| Best steady-state utilization | No | Better | Best |
| Closest to production engines | No | Partly | Yes |

---

## 4.11  What to Measure Honestly

Because this course keeps a slow control/data path on purpose, do not reduce Unit 4 to one throughput number.

Track at least four things:

- `compute utilization` - how often the MAC datapath is doing useful work
- `control ownership` - how many instructions or states exist per output element
- `working-set fit` - which tensors fit on chip and which require tiling
- `correctness` - whether the result matches a Python or NumPy reference

If you want one conceptual distinction to keep, make it this:

- `compute-phase utilization` tells you whether the engine itself is efficient
- `end-to-end utilization` tells you whether the whole system is dominated by launch and transfer overhead

Both numbers are useful. They answer different questions.

---

## 4.12  Checkpoint

By the end of this unit, you should be able to say "yes" to most of these, even if some parts are still design work:

- [ ] I can explain the difference between a datapath and an execution engine
- [ ] I can compare CPU-as-data-pump, CPU-as-sequencer, and hardware-sequencer designs
- [ ] I know what local stores the engine needs and why
- [ ] I can define a minimal execution contract for a 1x1 INT8 conv tile
- [ ] I can explain why `mac4_next` is a useful intermediate step
- [ ] I can explain why real accelerators usually move beyond a CPU in the hot loop
- [ ] I know what measurements matter besides raw end-to-end speed

---

## 4.13  Side Quests

- **DMA-like loader.** Add a tiny engine that copies host-visible buffers into local BSRAM without a CPU loop. This teaches why descriptor submission and DMA often arrive together.
- **Command queue.** Queue several descriptors so the device can process multiple tiles back-to-back. This is the simplest honest bridge from this course to real GPU command streams.
- **Ping-pong stores.** Double-buffer activations or filters so one bank loads while the other computes. This turns local memory from storage into a scheduling tool.
- **Microcode vs FSM.** Instead of a hard FSM, try a tiny microcoded controller ROM. Compare flexibility, area, and debuggability.
- **NVDLA comparison.** Read the public NVDLA docs and list three things it does that your current design does not. Decide which one matters most for learning.

Optional artifact ideas live in `public-artifacts.md`.

---

## 4.14  Deep Dives

- **CUDA Programming Guide, hardware implementation chapters** - for SMs, shared memory, warp scheduling, and launch behavior
- **Jouppi et al., TPU v1** - for fixed-function tensor compute and descriptor-like control
- **Chen et al., Eyeriss** - for dataflow choices and the cost of moving data
- **Sze et al., Efficient Processing of Deep Neural Networks** - the best broad survey of accelerator memory hierarchies and mapping strategies
- **NVDLA documentation** - for an open reference design that looks much closer to a shipping edge accelerator
- **CFU-Playground `hps_accel`** - for a nearby design space built on the same style of tooling
- **Roofline paper (Williams, Waterman, Patterson)** - for reasoning about compute-bound vs bandwidth-bound behavior
- **Jacob et al. quantization paper** - for the integer arithmetic conventions that make the post-processing path real instead of toy math

---

**Previous:** [Unit 3 — Kernel Fusion: Why Round Trips Kill You](03-fusion.md)
**Next:** [Unit 5 — The Compiler](05-compiler.md)
