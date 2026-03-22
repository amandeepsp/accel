# Unit 5: The Compiler — Graph Partitioning, Fallback, and Hardware Contracts

> **Series:** [00-architecture](00-architecture.md) → [01-compute](01-compute.md) → [02-datapath](02-datapath.md) → [03-fusion](03-fusion.md) → [04-engine](04-engine.md) → **[05-compiler](05-compiler.md)** → [06-model](06-model.md) → [07-systolic](07-systolic.md) → [08-feeding](08-feeding.md) → [09-modern](09-modern.md) → [10-redesign](10-redesign.md)

When you write `C = A @ B` in PyTorch, the framework does not send matrix math directly to the GPU. It lowers that math through several representations until it becomes something the hardware can execute.

This unit asks the compiler question for your accelerator: what exactly is the target, and which parts of a real graph should reach it at all?

Use tinygrad as the implementation path because it is small enough to read end to end. Use MLIR, TOSA, TVM, and Triton as comparison points that show how the same ideas scale in larger compiler stacks.

> **Status:**
> - `Implemented in repo:` instruction-level protocol and CFU primitives in `shared/protocol.zig`, `shared/cfu.zig`, `firmware/dispatch.zig`, `driver/driver.zig`, `hardware/top.py`
> - `Design exercise:` a backend that recognizes supported patterns and maps them to your hardware contract
> - `Stretch:` a descriptor-driven backend for a future Unit 4 engine with local stores, FIFO, and sequencer

The main lesson is not "write a backend because backends are cool." It is this: compilers only make sense when they know what hardware contract they are targeting, what should fall back, and where the graph should be partitioned.

---

## 5.1  What The Compiler Is Really Deciding

Every ML compiler eventually has to answer five questions:

- what operation is this, really?
- can my hardware run it?
- how should data be laid out for that hardware?
- what metadata does launch need?
- when should I fall back to another device?

That is true for tinygrad, XLA, TVM, Triton, TensorRT, and your accelerator.

```text
User math -> graph -> schedule -> IR -> target-specific representation -> launch contract
```

For a GPU, the launch contract is machine code plus grid/block dimensions.
For a fixed-function accelerator, the launch contract is often a descriptor plus data movement plus a command.

> **MLSys Connection:** This is the real compiler/hardware co-design loop. Hardware decides what contracts are cheap to execute. The compiler decides what programs can be expressed in those contracts. The boundary between them is where accelerator design becomes interesting.

---

## 5.2  The tinygrad Pipeline in One Pass

tinygrad is small enough that you can see the whole lowering pipeline without reading a million lines of code.

```text
User code:       C = A @ B
                    |
Lazy graph:      record the computation
                    |
Schedule:        group work into executable kernels
                    |
Linearize:       convert to UOps
                    |
Render:          target-specific representation
                    |
Compile:         binary artifact or callable object
                    |
Launch:          move data, execute, read results
```

The names vary across frameworks, but the structure is stable.

- `Lazy graph` means defer execution until you know more context.
- `Schedule` means choose execution boundaries.
- `Linearize` means express the work in a lower-level IR.
- `Render` means map that IR to the shape your target understands.
- `Compile` means package the result into something launchable.
- `Launch` means satisfy the runtime contract.

> **MLSys Connection:** XLA calls the IR HLO. Triton uses TTIR and TTGIR. TVM uses Relay and TIR. MLIR uses dialect stacks. tinygrad's UOps are simpler, but the ideas transfer directly.

---

## 5.3  Your Two Real Target Contracts

The most important design correction for this unit is to admit that your repo does not have just one meaningful compiler target.

It has two.

### Target A: The repo you have today

Today the hardware contract is instruction-level.

- `mac4`
- `srdhm`
- `rdbpot`
- accumulator read/reset
- `input_offset` register writes

This is exposed through:

- `shared/cfu.zig`
- `shared/protocol.zig`
- `firmware/dispatch.zig`
- `driver/driver.zig`

This target is useful for learning because it is real and executable right now.

### Target B: The engine you are designing

The future Unit 4 target is engine-level.

- load weights into local stores
- load activations into local stores
- load per-channel params
- send a descriptor
- execute
- drain buffered output

This target is useful for learning because it is how fixed-function accelerators actually become compiler targets.

### Why this split matters

If you skip straight to the future engine, the unit becomes speculative. If you only target today's instruction interface, the unit misses the main accelerator lesson.

So treat Unit 5 as a ladder:

1. inspect tinygrad and understand the backend contract
2. connect tinygrad to the current repo in the simplest possible way
3. design the cleaner descriptor-driven contract you eventually want

---

## 5.4  The 4-Part Device Contract

Every tinygrad backend has some version of four components.

### Allocator - where does data live?

The allocator decides what a "device buffer" means.

| Backend | Where buffers live |
|---|---|
| CUDA | GPU memory |
| Metal | `MTLBuffer` |
| CPU | host memory |
| Your current backend | host-side arrays until launch |

For your system, this is an honest simplification:

- the allocator can keep data in host memory
- the runtime moves only what is needed when launch happens
- local BSRAM is not general device memory; it is launch-time working memory

This is closer to a staging allocator than a conventional GPU memory model.

### Renderer - what is the "program"?

The renderer turns UOps into a target-specific representation.

| Target | Rendered form |
|---|---|
| CUDA | PTX or generated CUDA-like code |
| CPU | C or native callable form |
| Current repo target | a supported-op decision plus a call path |
| Future engine target | a compute descriptor |

For your future engine, the descriptor is the program because the inner instruction stream is fixed in hardware.

For today's repo, the renderer might be much simpler:

- recognize a supported pattern
- create a small metadata object
- route execution into a Python or Zig wrapper that talks to the current protocol

### Compiler - how does that representation become launchable?

For a GPU backend, compile often means invoking a real toolchain.

For your system, compile could mean any of these:

- no-op packaging around a Python callable
- packing a descriptor struct
- serializing launch metadata to bytes

The important lesson is not toolchain complexity. It is that there is always a boundary where target-specific representation becomes runtime-consumable.

### Runtime - how is execution triggered?

The runtime must:

- gather the concrete buffers
- move data into the right form
- launch supported work
- read results back
- surface errors and fallback cleanly

For today's repo, that runtime can still be instruction-oriented.
For the engine target, it becomes descriptor-oriented.

> **MLSys Connection:** Real accelerator stacks often look exactly like this: allocator, codegen, compile/package, runtime launch. The pieces have different names, but the separation of responsibilities is very stable.

---

## 5.5  What Should Be Lowered?

Not every op belongs on your accelerator.

That is not a weakness. It is normal.

The lowering decision is:

```text
offload when the hardware contract is both legal and worthwhile
```

Legality asks:

- does the hardware support the dtypes?
- does it support the operation shape?
- does it support the needed post-processing?
- does it fit the working-set constraints?

Worthwhile asks:

- is there enough compute density to justify movement and launch?
- will fallback boundaries create more traffic than they save?
- is this path simpler to validate on CPU?

For this course, start with a narrow supported set:

- INT8 dot product / tiny matmul
- maybe 1x1 conv expressed as matmul
- fused integer post-processing

Leave everything else on CPU at first.

Viewed at graph level, this is partitioning:

- mark the subgraphs that are legal and worthwhile for the accelerator
- keep unsupported or low-value regions on CPU
- try to avoid tiny accelerator islands that create more transfer cost than they save

That is why compiler support for accelerators is rarely just code generation. It is also placement and boundary management.

### Real-world comparison

- GPUs support many more operators directly, but compilers still pattern-match heavily.
- TensorRT, XLA, TVM, and ONNX Runtime all have legality checks and fallback paths.
- Fixed-function NPUs are even stricter: many subgraphs run on the accelerator, the rest stay elsewhere.

Teaching the supported subset explicitly is more honest than pretending backends are all-or-nothing.

---

## 5.6  Exercise 5a: Read tinygrad's small backends

Read tinygrad's simplest backends and map them onto the 4-part contract.

Suggested files:

- `tinygrad/runtime/ops_npy.py`
- `tinygrad/runtime/ops_cpu.py`

Questions:

- where is allocation handled?
- where is rendering handled?
- where is compilation handled?
- where is launch handled?
- which backend has the cleanest minimal path you can imitate first?

The goal is not to memorize tinygrad internals. The goal is to recognize the compiler/runtime split in a small codebase.

---

## 5.7  Exercise 5b: Trace one dot product through the pipeline

Run:

```bash
DEBUG=4 python -c "from tinygrad import Tensor; print(Tensor([1,2,3]).dot(Tensor([4,5,6])).item())"
```

Then answer:

- where does lazy evaluation end?
- where are schedule items formed?
- where do UOps appear?
- what information would your hardware backend need from that trace?
- at which stage would your backend first diverge from the CPU or CUDA paths?

This exercise matters because it turns "compiler backend" from a mysterious box into a visible sequence of transforms.

---

## 5.8  Exercise 5c: Choose an integration strategy

Do not assume the full backend is the right first step.

There are three reasonable paths.

| Path | What it proves | Why choose it |
|---|---|---|
| Manual Python driver | Host/runtime contract | Fastest way to validate launch semantics |
| `CUSTOM_FUNCTION`-style hook | Framework integration | Great middle step before a real backend |
| Full backend | End-to-end compiler target | Best architectural endpoint |

Recommended order:

1. manual Python path
2. tinygrad hook or interception path
3. full allocator/renderer/compiler/runtime backend

Why this is realistic:

- real compiler backends are developed incrementally
- runtime contracts often stabilize before codegen does
- hardware support and legality checks usually mature before full performance does

> **MLSys Connection:** This is how real accelerator enablement usually happens. First a validated runtime path. Then a narrow supported op set. Then more complete lowering and scheduling.

---

## 5.9  Exercise 5d: Build the simplest allocator

For your accelerator, the allocator can be intentionally boring.

```python
class ACCELAllocator(Allocator):
    def _alloc(self, size, options=None): ...
    def _copyin(self, dest, src): ...
    def _copyout(self, dest, src): ...
```

Key design choice:

- host memory is the durable storage
- launch-time code moves a working set into device-local storage or into the instruction path

That means the allocator is not where you express BSRAM layout. The runtime is.

Why this is educational:

- it forces you to distinguish storage ownership from execution working set
- it mirrors the difference between global memory and scratchpad/shared memory

---

## 5.10  Exercise 5e: Design the renderer honestly

The renderer is the hardest conceptual piece because it decides whether your hardware can express the work.

For this course, your renderer should answer two questions:

1. is this pattern supported?
2. if yes, what metadata does launch need?

For a first pass, a perfectly good renderer only recognizes:

- one reduction axis
- two INT8 inputs
- one INT32 accumulator or INT8 output path
- optional fused integer post-processing

Everything else should fall back.

### Two honest renderer targets

#### Renderer for today's repo

Emit a small object describing:

- operand roles
- shapes
- dtype checks
- which sequence of current protocol operations will run

This is not a beautiful production backend. It is an honest bridge from compiler IR to the system you actually have.

#### Renderer for the future engine

Emit a descriptor containing:

- operation kind
- `K`, `N`, `spatial`
- quantization parameters
- layout assumptions
- buffer role mapping

This is the cleaner fixed-function accelerator story.

### Why real accelerators do this

- renderers capture legality and target shape
- fixed-function accelerators benefit from compact descriptors
- programmable targets emit instructions because the hardware contract is broader

---

## 5.11  Exercise 5f: Build the runtime launch path

The runtime is where abstract metadata meets actual bytes.

For the current repo, one launch path might look like:

```text
buffers on host
  -> pack operands or tiles
  -> send request(s) over current protocol
  -> execute CFU-visible operations
  -> collect outputs
  -> reshape / store result
```

For the future engine target, one launch path might look like:

```text
buffers on host
  -> load weights
  -> load activations
  -> load per-channel params
  -> send descriptor
  -> execute
  -> drain result buffer or FIFO
```

In both cases the runtime must also answer:

- how are logical buffers mapped to roles?
- what happens when a transfer fails?
- how do you validate returned shape and count?
- what is the error path when hardware refuses an op?

This is where Units 2 and 4 meet the compiler story directly.

---

## 5.12  Exercise 5g: Handle unsupported ops and fallback

Fallback is not an admission of failure. It is part of the design.

For unsupported patterns:

- run on CPU
- keep the interface stable
- make the boundary visible in logs or profiling

Questions worth answering:

- should fallback happen per op or per subgraph?
- when does bouncing between CPU and accelerator cost too much?
- how do you make legality failures debuggable?

This is one of the most transferable lessons in the whole course. Real accelerator deployment is mostly partitioning and boundary management.

---

## 5.13  Stretch: Add profiling

Profile at least four phases:

- render / pattern-match time
- transfer time
- compute time
- drain / deserialize time

For your current repo, transfer and control overhead will dominate. That is fine.

The point is not to prove the accelerator is fast. The point is to learn what a backend can see and what it cannot hide.

---

## 5.14  The Incremental Path

Do the backend in stages.

```text
Stage 1: manual host script
  proves protocol, data packing, correctness

Stage 2: tinygrad hook or interception path
  proves framework integration and fallback boundary

Stage 3: narrow backend for one supported pattern
  proves renderer/runtime contract

Stage 4: descriptor-driven engine backend
  proves a true fixed-function compiler target

Stage 5: model-level integration
  proves operator placement, tiling, and real graph handling
```

This sequence is pedagogically better than pretending a full backend appears in one leap.

---

## 5.15  Checkpoint

By the end of this unit, you should be able to say yes to most of these, even if your backend is still partial:

- [ ] I can explain the difference between allocator, renderer, compiler, and runtime
- [ ] I can trace a tinygrad dot product through the pipeline to the point where my backend would take over
- [ ] I know the difference between targeting today's instruction-level repo and a future descriptor-driven engine
- [ ] I can define a narrow supported pattern for my hardware backend
- [ ] I can explain when and why fallback should happen
- [ ] I have at least one working path from framework or Python host code to current hardware execution
- [ ] I know what metadata a future engine descriptor would need

At full completion, a nice end-to-end test is:

```bash
DEVICE=ACCEL python -c "from tinygrad import Tensor; print(Tensor([1,2,3]).dot(Tensor([4,5,6])).item())"
```

Expected output: `32`

Treat that as a capstone, not as the minimum bar for learning from this unit.

---

## Side Quests

- **micrograd or pure-Python path first.** Build the smallest possible compiler/runtime integration before tinygrad.
- **ONNX Runtime comparison.** Read how execution providers partition graphs and compare that to your fallback strategy.
- **Descriptor schema design.** Write three versions of a descriptor: minimal, explicit, and future-proof. Compare simplicity vs extensibility.
- **Legality checker.** Build a pass that prints exactly why an op could not be lowered. This is one of the most practical compiler tools you can write.
- **Engine vs instruction backend.** Implement both a narrow current-repo target and a cleaner future-engine target on paper. Compare what the renderer and runtime have to know.

Optional artifact ideas live in `public-artifacts.md`.

If you want one partial implementation sketch, see `docs/course/solutions/05-compiler/ops_accel.py`.

---

## Suggested Readings

1. **tinygrad source** - start with `tinygrad/runtime/ops_npy.py`, `tinygrad/runtime/ops_cpu.py`, and the scheduling / UOps code paths.
2. **Triton paper** - to compare your descriptor-oriented target with a programmable compiler target.
3. **XLA overview** - for whole-graph lowering, placement, and runtime launch concepts.
4. **MLIR overview** - for how larger compiler stacks structure IR levels and legality boundaries.
5. **TVM docs and papers** - for operator selection, scheduling, and target annotations.
6. **TensorRT and ONNX Runtime docs** - for real-world fallback / partitioning concepts.
7. **NVDLA docs** - for an open fixed-function accelerator target where descriptors, buffers, and scheduling are explicit.
8. **Chris Lattner, "The Golden Age of Compiler Design in an Era of HW/SW Co-design"** - for the bigger picture.

---

**Previous:** [Unit 4 — The Execution Engine](04-engine.md)
**Next:** [Unit 6 — End-to-End: Run a Real Model](06-model.md)
