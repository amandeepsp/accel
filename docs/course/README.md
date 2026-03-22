# Course Roadmap - Learning ML Accelerators

This course is not a benchmark shootout and it is not a solution manual.

The current rewrite leans into a more opinionated, higher-learning-value version of the course: build something, discover why it is not enough, then redesign around the next real bottleneck.

The goal is to learn how ML accelerators are shaped by five recurring questions:

- where does data live?
- who owns control of the loop nest?
- what stays programmable and what becomes fixed-function?
- what bottleneck dominates now?
- what did this unit move off the CPU?

The repo is intentionally small and inspectable. Some parts are realistic in principle but tiny in scale. Some parts are deliberately unrealistic so the architecture is easier to see.

- The host/device split is real.
- Scratchpad-style local memory is real.
- Fused epilogues are real.
- Descriptor-driven engines are real.
- A 115200-baud UART standing in for PCIe is not realistic, but it makes control-plane cost impossible to ignore.

## How To Use This Course

Every unit should be read in five passes:

1. Predict what the bottleneck or tradeoff will be.
2. Build or sketch the mechanism.
3. Measure or inspect something concrete.
4. Explain why a real accelerator would or would not do it this way.
5. Optionally keep one artifact or note outside the main course text.

To keep the course honest, units use three status labels:

- `Implemented in repo` - backed by the current codebase.
- `Design exercise` - a serious architectural step that may not be implemented yet.
- `Stretch` - valuable extension once the core idea is clear.

## Current Repo Baseline

Today the repo already contains a working instruction-level accelerator path:

- Control plane: `shared/protocol.zig`, `driver/driver.zig`, `firmware/link.zig`, `firmware/dispatch.zig`
- CFU interface: `shared/cfu.zig`
- Hardware datapath: `hardware/top.py`, `hardware/mac.py`, `hardware/rw.py`, `hardware/quant.py`
- End-to-end verification: `hardware/test_top.py`

That means the course is currently strongest through Unit 3, and Unit 4 begins the transition from implemented datapath to designed execution engine.

## Learning Arc

The file numbering reflects how the repo evolved. The learning arc below is the more opinionated version of the course: each new unit is supposed to pressure-test the design you built before it.

## Unit Map

| Unit | Core Question | Status |
|---|---|---|
| `00-architecture.md` | Why do accelerators exist at all? | Implemented in repo |
| `01-compute.md` | Why is arithmetic only the beginning? | Implemented in repo |
| `02-datapath.md` | What does the host/device contract actually cost? | Implemented in repo |
| `03-fusion.md` | Why does moving bytes matter more than adding MACs? | Implemented in repo |
| `04-engine.md` | Which control plane should own execution: instructions, registers, descriptors, or an FSM? | Design exercise grounded in current repo |
| `05-compiler.md` | How does the compiler partition graphs, choose fallback, and target a hardware contract? | Design exercise / planned next step |
| `06-model.md` | What happens when a real quantized model meets your narrow accelerator contract? | Design exercise / planned next step |
| `07-systolic.md` | When does scaling compute help, and when does it just expose the memory wall faster? | Stretch / architectural capstone |
| `08-feeding.md` | How do real accelerators keep engines busy with DMA, queues, and overlap? | Design exercise / advanced systems module |
| `09-modern.md` | Why do modern workloads break CNN-friendly accelerator assumptions? | Design review / reality check |
| `10-redesign.md` | Given everything you learned, what accelerator would you build now? | Final architecture studio |
| `appendix-prior-art.md` | What do existing accelerators suggest we should change? | Reference |

## What Is New In The Wilder Version

- `Control planes` are now first-class, not hidden inside "engine" implementation details.
- `Graph partitioning and fallback` are treated as central compiler responsibilities, not side effects.
- `DMA, command queues, and async overlap` get their own module because feeding compute is architecture.
- `Modern workload reality checks` keep the course from stopping at a CNN-only worldview.
- The course now ends with a redesign studio instead of pretending the first architecture is the final one.

## Optional Public Artifacts

If you want to keep notes, diagrams, or shareable artifacts, keep them separate from the main course text. A lightweight template lives in `public-artifacts.md`.

The main course should stay focused on accelerator concepts, design choices, and implementation details.

## Resource Tracks

If you want to go deeper, follow a topic track instead of reading randomly.

- `Runtime and command submission`
  - CUDA Programming Guide, hardware implementation chapters
  - Fabien Sanglard, "How GPUs Work"
  - Fabian Giesen, "A Trip Through the Graphics Pipeline"
- `Quantization math`
  - Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
  - TFLite INT8 quantization docs
- `Dataflow and accelerator architecture`
  - Jouppi et al., TPU v1
  - Chen et al., Eyeriss
  - Sze et al., Efficient Processing of Deep Neural Networks
  - NVDLA documentation
- `Compiler and lowering`
  - tinygrad source
  - Triton paper
  - TVM docs and papers
  - MLIR and TOSA overviews
  - ONNX Runtime execution providers
- `Performance modeling`
  - Williams, Waterman, Patterson, Roofline paper
- `Edge models`
  - MobileNet v1/v2
  - MCUNet

## Design Principles For The Rewrite

- Prefer honest architectural comparisons over perfect analogies.
- Treat unrealistic bottlenecks as teaching tools, not bugs in the narrative.
- Make the learner explain choices before showing implementations.
- Keep side quests as first-class optional paths, not throwaway extras.
- Mark clearly what is implemented now, what is a design target, and what is a stretch.
