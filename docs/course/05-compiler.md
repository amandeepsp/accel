# Unit 5: The Compiler — Lowering IR to Your Hardware

> **Series:** [01-numbers](01-numbers.md) → [02-datapath](02-datapath.md) → [03-vertical](03-vertical.md) → [04-engine](04-engine.md) → **[05-compiler](05-compiler.md)**

When you write `C = A @ B` in PyTorch, dozens of compiler passes transform that into GPU machine code. tinygrad does the same thing in ~10K lines of Python. You're going to add a backend that targets your FPGA accelerator.

---

## 5.1  What Happens When You Write `A @ B`

Every ML framework follows the same pipeline. The user writes high-level math. The framework lowers it through successive stages until it becomes something hardware can execute:

```
  User code:       C = A @ B
                      |
  Lazy eval:       LazyBuffer — nothing happens yet
                      |
  Schedule:        Fuse ops into ScheduleItems
                      |
  Linearize:       Convert to UOps (micro-operations)
                      |
  Render:          UOps --> target code
                   CUDA --> PTX text
                   Metal --> MSL text
                   YOUR BACKEND --> compute descriptor
                      |
  Compile:         PTX --> cubin (nvcc)
                   MSL --> metallib (Metal compiler)
                   YOUR BACKEND --> binary struct (trivial)
                      |
  Launch:          cuLaunchKernel()
                   YOUR BACKEND --> UART send + START
```

tinygrad makes this pipeline explicit and readable. Each stage is a Python function you can inspect with `DEBUG=` flags.

> **MLSys Connection:** This pipeline exists in every ML compiler: XLA (TensorFlow), Glow (PyTorch/Meta), TVM, Triton, MLIR-based stacks. The names change — XLA calls the IR "HLO," TVM calls it "TIR," Triton calls it "TTIR/TTGIR" — but the structure is identical. High-level ops become loops, loops become memory accesses and ALU instructions, instructions become target-specific code. Understanding this pipeline in tinygrad (where it's ~10K lines) teaches you the same concepts as XLA (where it's ~1M lines).

---

## 5.2  The tinygrad Pipeline in Detail

### Lazy Evaluation

```python
a = Tensor([1, 2, 3])       # no compute yet
b = Tensor([4, 5, 6])       # no compute yet
c = a.dot(b)                 # STILL no compute — just records the operation
print(c.item())              # NOW it computes (forces realization)
```

tinygrad builds a graph of deferred operations. Nothing executes until a value is needed. This is the same strategy as PyTorch's `torch.compile`, JAX's tracing, and TensorFlow's graph mode.

Why defer? Because the framework can see the *full computation graph* before deciding how to execute it. This enables fusion — combining multiple operations into a single kernel launch.

### Schedule

The scheduler examines the lazy graph and groups operations into **ScheduleItems** — each one becomes a single kernel launch. Operations that can be fused (e.g., matmul + bias + ReLU) become one ScheduleItem. Operations with dependencies become separate items with explicit data movement between them.

### Linearize

Each ScheduleItem is converted to **UOps** — tinygrad's micro-operation IR. UOps are low-level but target-independent: loops (`RANGE`), loads (`LOAD`), stores (`STORE`), arithmetic (`ALU`), accumulation (`REDUCE_AXIS`). This is the last representation before target-specific code.

### Render

The renderer converts UOps to target-specific code. For CUDA, this emits PTX assembly text. For CPU, this emits C code. For Metal, MSL. For your backend, this is where you pattern-match the UOps and emit a **compute descriptor** — the configuration struct your sequencer FSM needs.

### Compile

The compiler takes rendered output and produces a binary. For CUDA: `nvcc` compiles PTX to a `.cubin`. For CPU: `clang` compiles C to a `.so`. For your backend: pack the compute descriptor dict into a binary struct. This step is trivial for you because your "program" is a configuration, not instructions.

### Launch

The runtime sends the compiled program to hardware and executes it. For CUDA: `cuLaunchKernel()`. For your backend: send the descriptor and data over UART, write config registers, assert START, wait for DONE, drain the output FIFO.

---

## 5.3  The 4-Part Device Contract

Every tinygrad backend implements four components. Understanding this contract is the key to adding a new backend.

### Allocator — Where Does Data Live?

The allocator manages device memory. It handles allocation, host-to-device copies, and device-to-host copies.

| Backend | `_alloc()` | `_copyin()` | `_copyout()` |
|---|---|---|---|
| CUDA | `cuMemAlloc()` | `cuMemcpyHtoD()` | `cuMemcpyDtoH()` |
| Metal | `MTLBuffer` | copy to `MTLBuffer` | copy from `MTLBuffer` |
| CPU | `malloc()` | `memcpy()` | `memcpy()` |
| **ACCEL** | **host-side numpy array** | **copy into numpy array** | **copy from numpy array** |

Your allocator is simple: data lives in host memory (numpy arrays) until it's needed on the FPGA. The UART transfer happens at launch time, not at allocation time. This is a valid strategy — some GPU backends also stage data in host memory and transfer on demand.

> **MLSys Connection:** Memory management is where real-world ML compilers spend enormous effort. CUDA's unified memory, Metal's shared memory, and AMD's hUMA all try to hide the host/device boundary. Your system makes the boundary explicit: data lives on the host until launch, then gets copied to BSRAM. This is actually closer to how discrete GPUs work (explicit `cuMemcpy`) than integrated GPUs (shared address space).

### Renderer — What Is the "Program"?

The renderer converts UOps to a target-specific representation. For traditional backends, this is source code (PTX, C, MSL). For your backend, it's a **compute descriptor** — a dictionary (or struct) containing the operation type, dimensions, quantization parameters, and data layout.

| Backend | Rendered Output | Example |
|---|---|---|
| CUDA | PTX assembly text | `mad.lo.s32 %r3, %r1, %r2, %r3;` |
| CPU | C source code | `out[i] = a[i] * b[i];` |
| **ACCEL** | **Compute descriptor dict** | `{"op": "matmul", "K": 8, "N": 8, ...}` |

Why is a descriptor valid as a "program"? Because your sequencer FSM has a fixed instruction stream. The "program" is the configuration — dimensions, addresses, quantization parameters — not the instructions themselves. The instructions are baked into the hardware.

> **MLSys Connection:** This is the same distinction between a **programmable** processor (GPU SM, CPU) and a **configurable** accelerator (TPU matrix unit, your sequencer). On a GPU, the renderer must emit actual instructions. On a configurable accelerator, the renderer emits parameters. Google's Edge TPU compiler does exactly this: it takes a TFLite graph and emits a sequence of *configuration descriptors* for the fixed-function hardware, not a stream of instructions.

### Compiler — From Representation to Binary

The compiler takes the rendered output and produces a binary artifact that the runtime can execute.

| Backend | Input | Output | Tool |
|---|---|---|---|
| CUDA | PTX text | `.cubin` binary | `nvcc` / `ptxas` |
| CPU | C text | `.so` shared library | `clang` |
| **ACCEL** | **descriptor dict** | **packed binary struct** | **`struct.pack()`** |

Your "compiler" is nearly trivial — serialize a Python dict into a binary struct. But it's conceptually the same step: transform a human-readable representation into something the hardware can consume.

### Runtime — Launch and Wait

The runtime sends the compiled program to hardware and manages execution.

| Backend | Launch | Wait | Read Results |
|---|---|---|---|
| CUDA | `cuLaunchKernel()` | `cuStreamSynchronize()` | `cuMemcpyDtoH()` |
| **ACCEL** | **UART send + START** | **poll DONE register** | **drain output FIFO over UART** |

Your launch sequence:
1. Send weight data over UART -> firmware loads into filter BSRAM
2. Send activation data over UART -> firmware loads into activation BSRAM
3. Send config registers over UART -> firmware writes sequencer config
4. Send START command -> firmware asserts START signal
5. Poll for DONE -> firmware reads DONE status
6. Drain output FIFO over UART -> firmware reads FIFO words and sends back

---

## 5.4  What About Ops You Can't Do?

Your hardware does INT8 matrix multiply with fused requantization. That's it. What about ReLU? Bias add? Reshape? Softmax? Float operations?

This is a real problem that every accelerator faces. No hardware supports every operation. The solution: **mixed-device execution**.

tinygrad (and every other framework) handles this with fallback. When an operation can't run on the target device, it runs on the CPU instead. The data moves from device to host, the CPU computes, and the result moves back.

```
  Mixed execution for: conv -> relu -> conv

  ACCEL: conv1 (INT8 matmul + requant)
           |
           v  [device-to-host copy]
  CPU:   relu (trivial, not worth offloading)
           |
           v  [host-to-device copy]
  ACCEL: conv2 (INT8 matmul + requant)
```

The data movement cost of bouncing between devices is real. This is why GPU compilers try so hard to fuse operations — every unfused op is a potential device boundary.

> **MLSys Connection:** ONNX Runtime calls these "execution providers" and routes ops to different devices. TensorRT marks ops it can handle and falls back to CUDA/CPU for the rest. XLA has a "host compute" mechanism for ops that can't run on TPU. The pattern is universal: accelerators handle the heavy compute, CPUs handle everything else, and the compiler minimizes data movement between them.

---

## 5.5  The Lowering Decision

Not every operation benefits from offloading. The math is simple:

```
  Offload when:
    Time_on_host > Time_serialize + Time_transfer + Time_execute + Time_deserialize
```

For your system with UART (~11,500 bytes/sec):

| Operation | Offload? | Why |
|---|---|---|
| Large INT8 matmul | Yes | Many MACs per byte transferred |
| Small matmul (<64 elems) | No | Setup overhead > compute time |
| Element-wise add/mul | No | 1 op per byte — UART kills you |
| Anything needing float | No | Hardware only does INT8 |
| ReLU | No | Trivial on host |
| Requantization | Yes | Fused into MAC pipeline (free) |

The UART bottleneck makes the crossover point unfavorable for most operations right now. But this is the wrong way to think about it:

1. **UART is a placeholder.** SPI gives ~10 MB/s. A custom parallel bus gives more. PCIe gives GB/s.
2. **With BSRAM stores, weights load ONCE per layer.** The amortized cost per output element drops dramatically.
3. **The architecture you're learning IS the GPU architecture.** The UART speed is irrelevant to the educational goal.
4. **For batch inference, activations stay on-device between layers.** Only the first input and last output cross the bus.

The real question isn't "is UART fast enough?" — it's "which operations have enough compute density to justify the transfer cost at your system's bandwidth?"

---

## 5.6  Exercises

### Exercise 5a: Read tinygrad's minimal backends

Read these files in the tinygrad source (install with `pip install tinygrad`):

1. `tinygrad/runtime/ops_npy.py` — The simplest possible backend. ~11 lines. It only stores data; it can't compute. Understand `NpyAllocator._alloc`, `_as_buffer`, `_copyout`.

2. `tinygrad/runtime/ops_cpu.py` — A compute-capable backend that emits C code, compiles it with clang, and calls the resulting `.so`. Find where the C renderer is selected, where clang is invoked, and where the compiled library is loaded and called.

Map each component to the 4-part contract: which class is the Allocator? The Renderer? The Compiler? The Runtime?

<details><summary>Hint 1</summary>

In `ops_npy.py`, the `NpyDevice` class inherits from `Compiled` and passes `allocator=NpyAllocator()`. There's no renderer, compiler, or runtime — NPY is a storage-only device. tinygrad uses it for loading model weights from `.npy` files.

</details>

<details><summary>Hint 2</summary>

In `ops_cpu.py`, look for the `CPUDevice` class. It wires together: `ClangRenderer` (renderer), `ClangCompiler` (compiler), `ClangProgram` (runtime), and `MallocAllocator` (allocator). The compiler invokes `clang` to compile C source to a `.so`, and the runtime loads it with `ctypes`.

</details>

<details><summary>Solution</summary>See solutions/05-unit/backend_analysis.md</details>

### Exercise 5b: Trace a dot product with DEBUG=4

Run this and study the output:

```bash
DEBUG=4 python -c "from tinygrad import Tensor; print(Tensor([1,2,3]).dot(Tensor([4,5,6])).item())"
```

Map each line of debug output to a pipeline stage:
- Where does lazy evaluation end and scheduling begin?
- Where are UOps generated?
- Where is code rendered?
- Where is it compiled and launched?

For your backend, which stage would produce the compute descriptor instead of C/PTX code?

<details><summary>Hint 1</summary>

With `DEBUG=4`, tinygrad prints the UOps for each kernel. Look for lines showing `RANGE`, `LOAD`, `ALU`, `REDUCE_AXIS`, and `STORE`. These are the UOps that your renderer would pattern-match. A dot product has a `RANGE` loop over the reduction axis, `LOAD`s for both inputs, a `MUL` + `ADD` (or `MULACC`), and a `STORE` for the output.

</details>

<details><summary>Hint 2</summary>

The render stage is where your backend diverges. Instead of emitting C code like `for (int i=0; i<3; i++) acc += a[i]*b[i];`, your renderer would recognize the pattern "RANGE over reduction axis + LOAD + MUL + ADD + STORE with int8 dtypes" and emit a compute descriptor: `{"op": "matmul", "K": 3, "N": 1}`.

</details>

<details><summary>Solution</summary>See solutions/05-unit/trace_analysis.md</details>

### Exercise 5c: Implement `ACCELAllocator`

Build the allocator for your backend. Data lives in host-side numpy arrays until launch time.

```python
class ACCELAllocator(Allocator):
    def _alloc(self, size, options=None):
        """Allocate a host-side byte buffer."""
        ...
    def _copyin(self, dest, src: memoryview):
        """Copy data into the host-side buffer."""
        ...
    def _copyout(self, dest: memoryview, src):
        """Copy data from the host-side buffer."""
        ...
```

This is the simplest component. The key design decision: data stays on the host until the runtime sends it to the FPGA. There is no "device memory" in the GPU sense — your BSRAM is managed by the runtime at launch time, not by the allocator.

<details><summary>Hint 1</summary>

The allocator just manages numpy byte arrays. `_alloc` returns a `numpy.empty(size, dtype=numpy.uint8)`. `_copyin` copies from source memoryview into the numpy array. `_copyout` copies from the numpy array into the destination memoryview.

</details>

<details><summary>Hint 2</summary>

Study `MallocAllocator` in `tinygrad/runtime/allocator.py` for the base class interface. Your implementation is even simpler because you don't need to manage device pointers or GPU memory pools — it's just numpy arrays on the host.

</details>

<details><summary>Solution</summary>See solutions/05-unit/allocator.py</details>

### Exercise 5d: Implement `ACCELRenderer`

This is the hardest part. The renderer must examine the UOps for a scheduled kernel and decide: can my hardware do this? If yes, emit a compute descriptor. If no, signal that this kernel should fall back to CPU.

Your renderer should pattern-match for:
- INT8 matrix multiply (dot product, matmul, conv2d as im2col + matmul)
- Fused requantization (bias + SRDHM + RDBPOT + clamp)
- Shapes that fit in your BSRAM budget

Everything else falls back to CPU.

<details><summary>Hint 1</summary>

Start by not implementing a real renderer at all. Instead, implement it at the `CUSTOM_FUNCTION` level — register a Python function that takes numpy arrays, calls your firmware over UART, and returns the result. This bypasses the render/compile pipeline entirely and lets you test the end-to-end flow. Graduate to a real renderer once the basic flow works.

</details>

<details><summary>Hint 2</summary>

If you do implement a renderer, the key is recognizing the UOp pattern for a matrix multiply: a `REDUCE_AXIS` (or `RANGE` loop) over the K dimension, with `LOAD`s from two buffers, a `MUL`, and an accumulation. Check that the dtypes are `dtypes.char` (INT8). If so, emit a descriptor dict with the dimensions extracted from the UOp shapes.

</details>

<details><summary>Hint 3</summary>

tinygrad's renderer base class has a method that receives the linearized UOps. You can iterate over them, checking op types and dtypes. For a first pass, just check: is this a single reduction over two INT8 input buffers producing one INT32/INT8 output? If yes, emit a descriptor. If anything else, raise `NotImplementedError` to trigger fallback.

</details>

<details><summary>Solution</summary>See solutions/05-unit/renderer.py</details>

### Exercise 5e: Implement `ACCELProgram.__call__()`

The runtime's `__call__` method is where the rubber meets the road. It takes a compiled program (your binary descriptor) and the buffer arguments, then executes on hardware.

The sequence:
1. Serialize weight data from the host buffer
2. Send over UART -> firmware loads into filter BSRAM
3. Serialize activation data from the host buffer
4. Send over UART -> firmware loads into activation BSRAM
5. Send config (dimensions, quant params) -> firmware writes sequencer registers
6. Send START command
7. Wait for DONE response
8. Drain output FIFO over UART
9. Deserialize result into the output buffer

This connects directly to the firmware protocol from Unit 3 and the autonomous sequencer from Unit 4.

<details><summary>Hint 1</summary>

Use the serial link protocol you already built. The `OpType` enum from Unit 4 maps directly to the steps: `load_weights`, `load_acts`, `load_params`, `execute`, `read_output`. Each step is a serial link request/response pair.

</details>

<details><summary>Hint 2</summary>

The `__call__` method signature receives the compiled program (your descriptor struct) and a list of buffer arguments (the actual data). You need to match buffers to roles — which buffer is the weight matrix? Which is the activation? The descriptor (from the renderer) should include this mapping.

</details>

<details><summary>Hint 3</summary>

For error handling: what if the UART transfer fails? What if the firmware reports an error? What if the output FIFO has the wrong number of elements? Add assertions that compare expected output count (from the descriptor) to actual FIFO drain count. This catches shape mismatches early.

</details>

<details><summary>Solution</summary>See solutions/05-unit/runtime.py</details>

### Exercise 5f: Handle unsupported ops

Implement CPU fallback for operations your hardware can't do. When the renderer encounters an unsupported UOp pattern (anything that isn't INT8 matmul), execution should transparently fall back to the CPU backend.

<details><summary>Hint 1</summary>

tinygrad's multi-device scheduling handles this if you set things up correctly. If your renderer raises an appropriate exception for unsupported ops, the scheduler can route those ops to a different device. Study how tinygrad handles `CUSTOM_FUNCTION` fallback — when the custom function returns `None` or raises, the framework retries with the default backend.

</details>

<details><summary>Hint 2</summary>

A simpler approach: implement your device as a wrapper that checks each operation before dispatching. If it's a supported pattern, use your ACCEL runtime. If not, delegate to the CPU device. This is the "interception" pattern rather than the "full backend" pattern — less elegant but easier to get working.

</details>

<details><summary>Solution</summary>See solutions/05-unit/fallback.py</details>

### Exercise 5g (Stretch): Add profiling

Measure where time goes in your pipeline:

- **Render time:** How long does pattern matching and descriptor generation take?
- **Transfer time:** How long does UART data transfer take? (This dominates.)
- **Compute time:** How long does the FPGA take for the actual MAC operations?
- **Drain time:** How long to read back results?

Add timing instrumentation to `ACCELProgram.__call__()` and print a breakdown per "kernel launch." Compare transfer time to compute time — this tells you whether you're compute-bound or transfer-bound.

<details><summary>Hint 1</summary>

Use `time.perf_counter()` around each phase of the launch sequence. Print a table showing transfer bytes, transfer time, compute time (START to DONE), and drain time. The ratio of compute time to total time is your "kernel efficiency" — analogous to GPU kernel profiling with `nsys` or `ncu`.

</details>

<details><summary>Solution</summary>See solutions/05-unit/profiling.py</details>

---

## 5.7  The Incremental Path

Don't build the full backend at once. Each step is independently testable:

```
  Step 1: Manual Python script
  +-- Serialize a matmul by hand (numpy arrays -> bytes)
  +-- Send over UART using your serial link
  +-- Compare result to numpy.dot()
  +-- PROVES: protocol works, firmware works, math is correct

  Step 2: Python function wrapper
  +-- Wrap Step 1 in: accel_matmul(a, b) -> result
  +-- Add input validation (shape, dtype checks)
  +-- PROVES: the interface is clean

  Step 3: CUSTOM_FUNCTION hook
  +-- Register accel_matmul with tinygrad
  +-- Run: a.dot(b) where a, b are on "ACCEL" device
  +-- PROVES: tinygrad integration works

  Step 4: Full backend (Allocator + Renderer + Compiler + Runtime)
  +-- DEVICE=ACCEL works for INT8 matmul
  +-- Unsupported ops fall back to CPU
  +-- PROVES: you have a real accelerator backend

  Step 5: Run a real model
  +-- Single-layer INT8 conv via tinygrad
  +-- Compare output to host-side reference
  +-- PROVES: real workloads work
```

> **MLSys Connection:** This incremental approach mirrors how real backends are developed. NVIDIA didn't build CUDA in one pass — early GPUs had fixed-function shader pipelines (like your fixed sequencer), then programmable vertex/pixel shaders (partially configurable), then CUDA (fully programmable). Each step proved the concept before adding generality. tinygrad itself started with a single backend and added others incrementally.

---

## 5.8  Checkpoint

The definitive test:

```bash
DEVICE=ACCEL python -c "from tinygrad import Tensor; print(Tensor([1,2,3]).dot(Tensor([4,5,6])).item())"
```

Expected output: `32`

This means:
- [x] tinygrad recognized `ACCEL` as a valid device
- [x] Tensors were allocated via your `ACCELAllocator`
- [x] The dot product was scheduled and linearized to UOps
- [x] Your `ACCELRenderer` recognized the INT8 matmul pattern and emitted a descriptor
- [x] Your `ACCELCompiler` serialized the descriptor to binary
- [x] Your `ACCELProgram.__call__()` sent data over UART, launched the sequencer, and read back results
- [x] The result matches the expected value

You have built a compiler backend that lowers high-level tensor operations to your custom FPGA hardware. The same pipeline that targets NVIDIA GPUs now targets your Tang Nano 20K.

---

## Suggested Readings

1. **tinygrad source code** — The entire compiler pipeline in ~10K lines of Python. Start with `tinygrad/engine/schedule.py` (scheduling), `tinygrad/codegen/uopgraph.py` (UOp optimization), and `tinygrad/runtime/ops_cpu.py` (a complete backend).
   [https://github.com/tinygrad/tinygrad](https://github.com/tinygrad/tinygrad)

2. **Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations (Tillet et al., 2019)** — Triton takes a similar approach: high-level Python -> IR -> target code. The "tiled" abstraction maps to your BSRAM tile sizes. Understanding Triton helps you see how your descriptor-based approach relates to a fully programmable compiler.
   [https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

3. **XLA: Optimizing Compiler for Machine Learning** — Google's ML compiler for TPU and GPU. The "HLO" IR is analogous to tinygrad's UOps. XLA's "custom call" mechanism is how you'd add a new hardware target — directly comparable to your `CUSTOM_FUNCTION` approach.
   [https://www.tensorflow.org/xla](https://www.tensorflow.org/xla)

4. **The Deep Learning Compiler: A Comprehensive Survey (Li et al., 2020)** — Covers TVM, Glow, XLA, MLIR, and others. Section 3 on IR design and Section 5 on backend code generation are most relevant. Helps you see where your tiny backend fits in the landscape.
   [https://arxiv.org/abs/2002.03794](https://arxiv.org/abs/2002.03794)

5. **MLIR: Scaling Compiler Infrastructure for Domain Specific Computation (Lattner et al., 2021)** — The multi-level IR framework that underpins modern ML compilers. Understanding MLIR's "dialect" concept helps explain why tinygrad's UOps work: they're a single-level IR that's "low enough" for all targets. Your backend proves you can target custom hardware from this level.
   [https://arxiv.org/abs/2002.11054](https://arxiv.org/abs/2002.11054)

---

**Previous:** [Unit 4 — The Execution Engine](04-engine.md)
