# Unit 10: Redesign Studio — What Accelerator Would You Build Now?

> **Series:** [00-architecture](00-architecture.md) → [01-compute](01-compute.md) → [02-datapath](02-datapath.md) → [03-fusion](03-fusion.md) → [04-engine](04-engine.md) → [05-compiler](05-compiler.md) → [06-model](06-model.md) → [07-systolic](07-systolic.md) → [08-feeding](08-feeding.md) → [09-modern](09-modern.md) → **[10-redesign](10-redesign.md)**
> **Reference:** [Prior Art & Architecture Decisions](appendix-prior-art.md)

If the course worked, you should now be slightly dissatisfied with your first accelerator.

Good.

That is the point of this unit.

The goal is not to "finish the implementation." The goal is to synthesize what you learned and design a better accelerator on purpose.

> **Status:**
> - `Implemented in repo:` a narrow instruction-level accelerator path and the hardware/software building blocks you studied earlier
> - `Design exercise:` an intentional v2 architecture grounded in real workload and system constraints
> - `Stretch:` implement one chosen redesign axis in the repo

---

## 10.1  Why End With Redesign?

Because the wrong lesson from a project course is:

```text
I built one thing -> therefore that thing must have been the right architecture
```

The right lesson is:

```text
I built one thing -> now I understand why I would change it
```

Real accelerator teams iterate exactly like this.

- first architecture proves the basic contract
- second architecture fixes the most painful bottlenecks
- later architectures specialize or generalize based on target workload

This unit turns the course from a walkthrough into an architecture studio.

---

## 10.2  Three Honest V2 Directions

You do not need infinite options. Three archetypes are enough.

### A. Tiny Edge NPU

Primary goal:

- run a narrow set of quantized CNN-friendly operators efficiently

Typical traits:

- descriptor-driven fixed-function engine
- explicit local SRAM management
- DMA-like loader
- heavy fallback for unsupported ops

Best when:

- the workload is known
- power and area matter more than flexibility
- correctness and predictable deployment matter more than generality

### B. Tiny TPU-ish Matrix Engine

Primary goal:

- maximize dense matrix throughput for a narrow compiler contract

Typical traits:

- systolic or tensor-style array
- large local buffers relative to compute
- strong dependence on compiler tiling and scheduling
- less friendly to irregular ops

Best when:

- dense matmul-like workloads dominate
- compiler sophistication is acceptable
- the memory system can actually feed the array

### C. Tiny GPU-ish Accelerator

Primary goal:

- widen the set of supported kernels with a more programmable model

Typical traits:

- more expressive instruction or microcode stream
- scalar/vector helper paths alongside dense compute
- more complex control plane and runtime
- fewer hard assumptions baked into one descriptor type

Best when:

- workload diversity matters
- a single fixed-function contract is too limiting
- you are willing to spend more control complexity for flexibility

---

## 10.3  Architecture Scorecard

Use a scorecard instead of vibes.

| Axis | Questions to ask |
|---|---|
| Workload target | CNNs only? small models? transformer blocks? mixed workloads? |
| Control plane | custom instructions, MMIO, descriptors, microcode, FSM? |
| Memory hierarchy | where do weights, activations, params, and persistent state live? |
| Data movement | CPU loops, DMA, queued copies, overlap? |
| Compute style | SIMD MAC, array, vector helpers, mixed engines? |
| Compiler contract | one op at a time, tiled descriptor, queued graph fragment? |
| Fallback model | per op, per subgraph, explicit unsupported list? |
| Debuggability | can you still inspect and reason about it easily? |
| Resource budget | LUTs, DSPs, BSRAM, clock target, bandwidth target |

If your redesign does not answer those, it is not really a redesign yet.

---

## 10.4  Choose What To Optimize For

Every redesign starts by choosing a priority.

Pick one primary objective:

- `throughput`
- `simplicity`
- `energy / area efficiency`
- `compiler friendliness`
- `workload coverage`

Then name the sacrifice that comes with it.

Examples:

- optimize for throughput -> sacrifice simplicity
- optimize for workload coverage -> sacrifice area and control simplicity
- optimize for simplicity -> sacrifice peak performance and coverage

This matters because most bad architecture documents try to optimize everything at once.

---

## 10.5  Pick a Better Control Story

One of the biggest things this course should leave you with is that control planes are architectural.

Pick one of these as your v2 control story:

| Control story | Best for | Main cost |
|---|---|---|
| Custom instructions | tight coupling and easy bring-up | poor batching and high control overhead |
| MMIO registers + START | simple fixed-function engines | weak scalability |
| Descriptors + queue | efficient narrow engines | compiler/runtime must know more |
| Microcode | flexible semi-programmable engines | extra control complexity |
| Scalar core + accelerator blocks | diverse workloads | much more system design |

Then explain why you chose it for your workload target.

---

## 10.6  Pick a Better Memory Story

A redesign that only changes compute is usually incomplete.

Decide:

- what must stay on chip?
- what can stream?
- what needs persistence across launches or tokens?
- what should be double-buffered?
- what should be compiler-managed versus hardware-managed?

Good redesigns are often just better memory stories in disguise.

Examples:

- more activation SRAM to reduce tiling overhead
- explicit ping-pong stores for overlap
- parameter cache for reused weights
- persistent state region for sequence workloads
- wider local-memory ports so the array stops starving

---

## 10.7  Pick a Better Compiler Contract

After this course, a compiler target should no longer feel mystical.

For your v2 design, answer:

- what exactly is the unit of work submitted by software?
- what metadata is required?
- what layouts are assumed?
- what ops are guaranteed legal?
- what falls back?

### Three common contracts

#### Narrow explicit descriptor

- easiest to validate
- best for fixed-function engines
- worst for generality

#### Family of descriptors

- one for dense matmul / conv
- one for copies
- one for vector or reduction helper ops

This is often a sweet spot.

#### Tiny instruction or micro-op stream

- more flexible
- heavier compiler burden
- easier to grow toward generality

None of these is automatically right. The point is to choose consciously.

---

## 10.8  Exercises

### Exercise 10a: Write a one-page architecture thesis

Start with this sentence:

```text
I want to build a _____ accelerator for _____ workloads, and I am willing to trade away _____ to get it.
```

If you cannot fill that in, the rest will stay fuzzy.

### Exercise 10b: Draw your v2 block diagram

Include at minimum:

- host/runtime boundary
- control path
- local memory hierarchy
- compute blocks
- data movement engines
- output path

### Exercise 10c: Define the software contract

Write down the interface software sees:

- instructions?
- registers?
- descriptors?
- queues?
- completion signals?

### Exercise 10d: State what you will *not* support

Good architectures are defined as much by what they exclude as by what they include.

Write an explicit not-supported list.

### Exercise 10e: Choose one implementation spike

Pick one concrete next build step such as:

- DMA-like loader
- command queue
- local-store double buffering
- vector helper block
- better descriptor schema
- minimal microcode controller

The point is to end with a focused next move, not a vague dream.

---

## 10.9  Final Scorecard

By the end of this unit, you should be able to say:

- [ ] I know what workload my accelerator is really for
- [ ] I can justify my chosen control plane
- [ ] I can justify my chosen memory hierarchy
- [ ] I can describe the compiler/runtime contract in concrete terms
- [ ] I know what should fall back and why
- [ ] I know what one change would most improve the current repo next
- [ ] I can explain why my v2 design is better than my v1 design for its target workload

---

## Side Quests

- **Three-way redesign memo.** Write one page each for a tiny NPU-ish, TPU-ish, and GPU-ish version. Compare them.
- **Budget-first design.** Start from a hard resource budget: DSPs, BSRAMs, LUTs, clock, and I/O bandwidth. What architecture survives?
- **Compiler-first design.** Start from what tinygrad or MLIR would find easiest to target. What hardware does that imply?
- **Workload-first design.** Start from MobileNet, keyword spotting, or a toy transformer. What changes immediately?
- **Kill a sacred assumption.** Remove one assumption from your original design, like "CPU manages all data movement" or "only conv-like ops matter," and see what architecture falls out.

Optional artifact ideas live in `public-artifacts.md`.

---

## Suggested Readings

1. **NVDLA documentation** - for a concrete fixed-function edge-accelerator reference point.
2. **TPU v1 paper** - for a matrix-engine-first design philosophy.
3. **CUDA / GPU architecture docs** - for the programmable end of the design spectrum.
4. **Sze et al.** - for the most useful broad framework when comparing memory, dataflow, and area/energy tradeoffs.
5. **ONNX Runtime, TVM, MLIR, and XLA material** - for how compiler contracts influence hardware shape.

---

**Previous:** [Unit 9 — Modern Reality Check](09-modern.md)
