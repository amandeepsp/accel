# Unit 9: Modern Reality Check — When CNN-Friendly Designs Hit a Wall

> **Series:** [00-architecture](00-architecture.md) → [01-compute](01-compute.md) → [02-datapath](02-datapath.md) → [03-fusion](03-fusion.md) → [04-engine](04-engine.md) → [05-compiler](05-compiler.md) → [06-model](06-model.md) → [07-systolic](07-systolic.md) → [08-feeding](08-feeding.md) → **[09-modern](09-modern.md)** → [10-redesign](10-redesign.md)
> **Reference:** [Prior Art & Architecture Decisions](appendix-prior-art.md)

MobileNet and 1x1 convs are a good teaching target.

They are not the whole world.

This unit exists to puncture a very common misunderstanding:

```text
I built a decent CNN accelerator -> therefore I understand modern ML accelerators
```

Not quite.

Modern accelerator design is shaped by transformers, attention, KV cache traffic, layernorm, softmax, MLP blocks, and much larger working sets than your first edge-CNN design expects.

> **Status:**
> - `Implemented in repo:` a narrow instruction-level path that is naturally friendlier to quantized CNN-style dense compute than to transformer-style full-model execution
> - `Design exercise:` workload comparison, mismatch analysis, and extension proposals
> - `Stretch:` retarget the course architecture toward a different workload class

---

## 9.1  What Your Current Design Is Good At

Your current accelerator ideas line up naturally with a specific workload shape:

- dense INT8 compute
- fixed or simple loop nests
- strong data reuse across channels
- local post-processing after accumulation
- a modest working set that can be tiled into SRAM

That is why pointwise convolution feels so natural.

It has:

- predictable access patterns
- high MAC density
- small, stable per-output epilogues
- a nice mapping to packed dot products or small arrays

This is not an accident. Early accelerators and many edge NPUs are designed around exactly these properties.

---

## 9.2  Why Modern Workloads Feel Different

Now compare that with a transformer block.

Instead of mostly conv and epilogue, you get a mix like this:

- Q / K / V projections
- attention score matmuls
- softmax
- value mixing
- MLP layers
- residual adds
- layernorm
- sometimes KV cache reads and writes

The resulting pain points are different.

| Workload trait | CNN-friendly accelerator | Transformer reality |
|---|---|---|
| Dominant dense op | 1x1 conv / matmul | many matmuls, but mixed with memory-heavy ops |
| Control pattern | regular and static | still regular, but more phases and more buffers |
| Post-processing | per-channel epilogue | softmax, norm, masking, residual mixing |
| Working set pressure | tiles of activations and weights | activations plus long-lived KV cache |
| Layout assumptions | channel-major conv-like reuse | sequence-length and head-dimension dependent |

The point is not that transformers are chaotic. The point is that they stress different parts of the system.

---

## 9.3  The Four Biggest Mismatches

### 1. Softmax and normalization are not simple epilogues

Your course hardware loves local, channel-wise post-processing.

Softmax does not look like that.

- needs reductions
- needs exponent-like behavior or approximations
- needs normalization across a dimension, not just pointwise adjustment

Layernorm and RMSNorm also need statistics across vectors, not just one channel at a time.

### 2. KV cache turns memory into the problem

Autoregressive inference keeps past keys and values alive.

That means:

- long-lived state
- repeated reads of large buffers
- awkward reuse patterns compared with a local conv tile

The accelerator stops being "a box that consumes one tile and forgets it" and becomes "a system that manages persistent context."

### 3. Sequence length changes the economics

A CNN accelerator often assumes a relatively stable tiling story.

Attention cost scales with sequence length in ways that change:

- working-set size
- memory traffic
- arithmetic intensity
- queueing and latency behavior

This makes static, narrow assumptions less comfortable.

### 4. Modern models want mixed compute styles

Real ML accelerators rarely rely on only one compute primitive.

They mix:

- matrix engines or tensor cores
- vector/scalar units
- special function units or approximations
- flexible load/store and permutation hardware

That is the real lesson: a single fixed-function conv-friendly datapath teaches great fundamentals, but general model execution usually wants more than one kind of engine.

---

## 9.4  What Real Accelerators Actually Add

When hardware vendors had to support modern workloads, they did not just make the MAC array bigger.

They added a broader system.

### GPUs

- programmable instruction streams
- tensor cores for dense matrix math
- vector/scalar ALUs for control-ish and elementwise work
- large memory systems and sophisticated scheduling

### Edge NPUs / fixed-function accelerators

- descriptor-driven conv / matmul engines
- local SRAM and DMA
- a CPU or microcontroller nearby for unsupported ops
- graph partitioning in the runtime or compiler

### TPU-like designs

- very large matrix engines
- strong compiler/runtime coordination
- large on-chip buffers
- explicit data movement and tiling strategies

The shared theme is important:

```text
modern support comes from a richer system, not just a wider multiplier array
```

---

## 9.5  Three Honest Responses

Once you see the mismatch, there are only a few honest architectural responses.

### Response A: Stay specialized

Say clearly:

- this accelerator is for quantized CNN-like workloads
- fallback handles everything else

This is a perfectly valid strategy, especially for teaching and for edge devices.

### Response B: Add helper engines

Keep the main dense compute engine, but add support for:

- vector reductions
- norm-like operations
- simple activation or approximation units
- better persistent-buffer handling

This is how many real systems evolve.

### Response C: Move toward programmability

Accept that the workload mix is too broad for a narrow fixed contract.

That pushes you toward:

- more expressive descriptors
- microcode
- a tightly-coupled scalar core
- or a more GPU-like programming model

This is not always the right choice, but it is the direction generality pulls you.

---

## 9.6  Exercises

### Exercise 9a: Map a transformer block to your current accelerator

Take one transformer block and mark each sub-op as:

- natural fit
- awkward fit
- CPU fallback
- impossible without architectural change

You do not need to implement anything. The point is to see the pattern.

### Exercise 9b: Identify the first breaking point

Pick the first modern-workload feature that would seriously break your current design:

- softmax
- layernorm / RMSNorm
- KV cache
- long-sequence tiling
- mixed precision

Then explain *why* it breaks the current assumptions.

### Exercise 9c: Choose your philosophy

Choose one stance and defend it:

1. remain a narrow CNN / edge-NPU design
2. add helper engines for common transformer pain points
3. move toward a more programmable accelerator

### Exercise 9d: Write one extension proposal

Write a one-page proposal for one extension:

- new control primitive
- new memory structure
- new vector/reduction helper block
- persistent-cache mechanism

State clearly what workload it helps and what complexity it adds.

---

## 9.7  Checkpoint

- [ ] I can explain why a CNN-friendly accelerator is not automatically a good transformer accelerator
- [ ] I can name the biggest mismatch between my current design and modern workloads
- [ ] I can compare specialization, helper-engine, and programmability responses
- [ ] I understand why KV cache and normalization shift the bottleneck toward memory and mixed compute styles
- [ ] I can defend what workload class my accelerator should target next

---

## Side Quests

- **Attention on paper.** Write the buffer and loop structure for one small attention head and mark where your current hardware contract breaks.
- **Vector helper sketch.** Design a tiny vector reduction / norm helper and explain how software would use it.
- **Edge TPU comparison.** Read public Edge TPU or NPU material and list what they explicitly do *not* support.
- **Llama vs MobileNet.** Build a table of which assumptions hold for MobileNet but fail for a small LLM.
- **Persistence budget.** Estimate how much on-chip memory a tiny KV cache would really need at one toy scale.

Optional artifact ideas live in `public-artifacts.md`.

---

## Suggested Readings

1. **FlashAttention papers** - for how memory movement reshapes attention performance.
2. **Transformer inference optimization blogs and papers** - especially on KV cache bottlenecks.
3. **NVIDIA Tensor Core and Hopper architecture docs** - for mixed dense compute plus more general execution support.
4. **ONNX Runtime, TensorRT, and TVM docs** - for how modern graphs are partitioned across hardware capabilities.
5. **Sze et al.** - to compare classic accelerator dataflow ideas with newer workload demands.
6. **TinyML and edge-model papers like MCUNet** - for cases where staying specialized is exactly the right choice.

---

**Previous:** [Unit 8 — Feeding the Beast](08-feeding.md)
**Next:** [Unit 10 — Redesign Studio](10-redesign.md)
