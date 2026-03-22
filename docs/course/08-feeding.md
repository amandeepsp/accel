# Unit 8: Feeding the Beast — DMA, Command Queues, and Async Overlap

> **Series:** [00-architecture](00-architecture.md) → [01-compute](01-compute.md) → [02-datapath](02-datapath.md) → [03-fusion](03-fusion.md) → [04-engine](04-engine.md) → [05-compiler](05-compiler.md) → [06-model](06-model.md) → [07-systolic](07-systolic.md) → **[08-feeding](08-feeding.md)** → [09-modern](09-modern.md) → [10-redesign](10-redesign.md)
> **Reference:** [Prior Art & Architecture Decisions](appendix-prior-art.md)

You built arithmetic. Then local memory. Then a control path. Then maybe even a small array.

Now the next reality check arrives: the compute block is hungry, and software is a bad waiter.

This unit is about the systems machinery real accelerators use to keep compute busy:

- DMA-like loaders
- command queues
- doorbells and completion signals
- double buffering
- overlapping transfer, compute, and drain

> **Status:**
> - `Implemented in repo:` host-driven request/response protocol and explicit software-controlled transfers
> - `Design exercise:` DMA-like load engine, queued descriptors, and overlapped execution
> - `Stretch:` independent load / compute / drain engines with semaphores or events

---

## 8.1  Why This Module Exists

Once a compute engine gets faster, starvation becomes the dominant problem.

```text
slow engine + slow feed   -> compute dominates
fast engine + slow feed   -> transfer dominates
fast array + CPU copies   -> control path becomes the bottleneck
```

That is why real accelerators rarely stop at "a fast MAC" or even "a fast array." They add machinery whose whole job is to move data and launch work efficiently.

> **MLSys Connection:** GPUs have copy engines, command processors, streams, and multiple in-flight kernels. TPUs and NPUs use descriptor queues, DMA engines, and software-visible completion mechanisms. The tensor core or array gets the headlines; the feeding machinery determines whether it is actually busy.

---

## 8.2  The Three Feeding Models

Treat feed mechanisms as a ladder, just like control planes.

| Model | Who moves data? | Who launches work? | What you learn | Why real hardware uses or avoids it |
|---|---|---|---|---|
| CPU copy loop | CPU | CPU | Baseline visibility | Great for bring-up, terrible for scale |
| DMA + doorbell | DMA-like engine | CPU | Separation of movement from control | Common in real accelerators |
| Queued async engines | Load / compute / drain engines | Command queue | Throughput-oriented overlap | What high-performance systems converge toward |

### CPU copy loop

This is where small FPGA bring-up often starts.

- software writes every tile
- software launches every tile
- software waits and drains every tile

It is easy to debug because nothing is hidden.

It is also exactly what production systems try to escape.

### DMA + doorbell

The CPU still decides *what* should happen, but a dedicated engine handles *how bytes move*.

- CPU writes source, destination, length
- DMA copies host-visible memory into local stores
- CPU rings a doorbell or writes START
- CPU later observes DONE or gets an interrupt

This is the first serious systems step.

### Queued async engines

Now the CPU prepares work in advance.

- descriptor queue for future tiles
- load engine fills the next buffer
- compute engine works on the current one
- drain engine empties the previous output

This is how transfer latency gets hidden behind compute.

---

## 8.3  The Control Plane Meets the Data Plane

Unit 4 asked who owns the loop nest.
This unit asks a different question:

```text
who owns moving bytes into the right place at the right time?
```

That is the boundary between control plane and data plane.

| Concern | Control plane | Data plane |
|---|---|---|
| What to execute | descriptor or command | n/a |
| Where tensors go | addresses, lengths, formats | actual copied bytes |
| When work starts | doorbell / START / queue push | engine consumes prepared state |
| When work finishes | DONE / interrupt / queue completion | results written to buffer or FIFO |

Real accelerators spend a lot of design effort on this split because it lets software stay simple while hardware stays busy.

---

## 8.4  DMA Is More Than a Fast memcpy

In this course, a DMA-like engine is useful even if the transport is still slow.

Why?

- it teaches descriptor-driven movement
- it removes CPU loops from the transfer path
- it makes overlap possible later
- it creates a cleaner runtime contract for the compiler

### A minimal DMA descriptor

```text
src_addr      host-visible source buffer
dst_kind      activation / filter / param store
dst_offset    local-store offset
length_bytes  transfer size
flags         last_in_group? interrupt_on_done?
```

That is already enough to teach three big ideas:

1. software submits *intent*, not individual writes
2. movement can be decoupled from compute
3. descriptors are not just for compute engines; they are for copy engines too

### Real-world comparison

- NVIDIA: copy engines plus GPU command submission
- NVDLA: dedicated memory and convolution scheduling logic
- Edge NPUs: firmware-managed DMA into local SRAM
- network cards: descriptor rings that look surprisingly similar

---

## 8.5  Command Queues and Doorbells

Once descriptors exist, queues become natural.

```text
CPU prepares desc0, desc1, desc2
CPU pushes them into a ring buffer
CPU rings a doorbell
hardware processes until queue empty
```

This is one of the most important system concepts in accelerator design because it changes the host/device relationship:

- without a queue, software babysits every tile
- with a queue, software can run ahead

### Minimal queue design

| Field | Meaning |
|---|---|
| `head` | next descriptor for hardware to pop |
| `tail` | next slot software may fill |
| `count` | optional occupancy bookkeeping |
| `doorbell` | tells hardware new work is ready |
| `status` | idle / busy / queue empty / error |

### Why real accelerators use this

- amortize control overhead across many tiles
- allow burst submission from the host
- enable overlap with other software work
- make interrupts and events meaningful

> **MLSys Connection:** CUDA streams, GPU command buffers, submission queues in NPUs, NVMe submission/completion rings, and NIC descriptor queues are all family members of the same idea: software enqueues work, hardware drains it independently.

---

## 8.6  Async Overlap and Double Buffering

If there is one scheduling trick every accelerator student should know, it is this one:

```text
time ->

load tile 0     compute tile 0     drain tile 0
     load tile 1     compute tile 1     drain tile 1
          load tile 2     compute tile 2     drain tile 2
```

The whole point is to stop serializing the pipeline like this:

```text
load -> compute -> drain -> load -> compute -> drain
```

### Double buffering

The smallest practical form of overlap is ping-pong buffering:

- bank A is active for compute
- bank B is being filled for the next tile
- swap roles when compute completes

This turns local SRAM from passive storage into a scheduling tool.

### What must be true for overlap to help?

- the next tile is known early enough
- local memory is partitioned cleanly enough to avoid hazards
- control state can track which bank is live
- load time and compute time are close enough that overlap matters

### Why real hardware complicates this

Once overlap exists, correctness requires thinking about:

- ownership of each bank
- completion ordering
- backpressure when drain falls behind
- recovery when one engine faults

That is why asynchronous systems feel like a big jump: the throughput is better, but the state space is much larger.

---

## 8.7  What This Repo Could Build Next

For this project, a realistic next step is not "full GPU-style async everything."

It is a deliberately small version:

1. add a DMA-like copy engine for host-visible memory to local stores
2. add a small descriptor queue
3. add ping-pong filter or activation banks
4. add one completion mechanism: either polling or interrupt

That is enough to teach the essential systems ideas without drowning in complexity.

### A good staged path

```text
Stage 1: CPU loops write local stores
Stage 2: DMA-like loader fills local stores
Stage 3: queued compute descriptors
Stage 4: ping-pong buffering
Stage 5: load / compute overlap
Stage 6: load / compute / drain overlap
```

Do not skip straight to Stage 6.

---

## 8.8  Exercises

### Exercise 8a: Design a copy-engine descriptor

Write down the smallest descriptor that can load:

- an activation tile
- a filter tile
- a parameter block

Then answer:

- which fields are common?
- which are copy-type specific?
- which error states matter?

### Exercise 8b: Sketch a 4-entry command queue

Draw or describe:

- queue storage
- head/tail updates
- full/empty conditions
- what happens when the CPU rings a doorbell

### Exercise 8c: Plan one overlap timeline

Pick a tiled 1x1 conv and draw a timeline with three lanes:

- load
- compute
- drain

Show exactly where double buffering helps and where it does not.

### Exercise 8d: Decide what should interrupt software

Choose one:

- queue empty
- DMA complete
- output half full
- output full
- fatal fault

Explain why that interrupt is the most useful first one.

---

## 8.9  Checkpoint

- [ ] I can explain why fast compute alone does not create high throughput
- [ ] I can compare CPU copy loops, DMA-style movement, and queued async execution
- [ ] I can define a minimal copy descriptor
- [ ] I can define a minimal command queue
- [ ] I understand how double buffering creates overlap
- [ ] I can explain what state or hazards make async designs harder
- [ ] I can name the smallest version of this machinery worth building next in this repo

---

## Side Quests

- **Interrupt vs polling.** Decide whether your first completion path should be a status register or an interrupt. Defend the choice.
- **Queue occupancy counter.** Add a hardware-visible occupancy counter and think about when software should stop submitting work.
- **Software queue first.** Implement a host-side queue before building a hardware queue. What complexity disappears, and what does not?
- **Trace a real system.** Read how CUDA streams or ONNX Runtime execution providers schedule work and map the concepts to your own queue model.
- **Descriptor compression.** What fields are repeated often enough that you could factor them into a persistent context instead of sending them every time?

Optional artifact ideas live in `public-artifacts.md`.

---

## Suggested Readings

1. **CUDA Programming Guide** - streams, asynchronous copies, and execution overlap.
2. **NVDLA documentation** - especially the sections on memory movement and scheduling.
3. **Hennessy and Patterson** - DMA and memory-system chapters for classic control/data plane ideas.
4. **ONNX Runtime execution providers** - a software view of queued work and accelerator handoff.
5. **CFU-Playground and LiteX DMA-like components** - for nearby open-source examples.
6. **Sunburst / FIFO design notes** - for thinking about buffering and producer-consumer decoupling.

---

**Previous:** [Unit 7 — Spatial Parallelism: Systolic Arrays](07-systolic.md)
**Next:** [Unit 9 — Modern Reality Check](09-modern.md)
