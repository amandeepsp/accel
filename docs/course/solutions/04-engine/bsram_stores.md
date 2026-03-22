# BSRAM Stores with Prefetch Counters

## Architecture

The BSRAM stores are just `amaranth.Memory` instances (synthesize to Gowin BSRAM).
The "prefetcher" is two address counters + two comparators — no holding registers
needed because the SRAM's internal output register acts as the buffer.

```
  Write path (firmware-driven):           Read path (prefetch):
  ┌──────────────┐                        ┌──────────────┐
  │ WriteRegs    │                        │ filt_rptr    │──→ filt_mem.r_addr
  │ funct7=3     │──→ filt_mem.w_addr     │ (auto-inc    │
  │ in0=addr     │    filt_mem.w_data     │  on consume, │    filt_mem.r_data ──→ MAC mux
  │ in1=data     │    filt_mem.w_en       │  wraps at    │
  └──────────────┘                        │  k_count)    │──→ first, last signals
                                          └──────────────┘
```

Write pointers live in firmware (CPU passes explicit addresses via `in0`).
Read pointers live in hardware (5 signals + 2 comparators total).

---

## Hardware State (what lives in the CFU)

```python
# Filter store
filt_mem  = Memory(shape=32, depth=512, init=[])   # 2 KiB BSRAM
filt_rptr = Signal(16)                              # read address counter
k_count   = Signal(16)                              # cyclic wrap point (K/4)
first     = Signal()                                # filt_rptr == 0
last      = Signal()                                # filt_rptr == k_count - 1

# Activation store
act_mem   = Memory(shape=32, depth=512, init=[])   # 2 KiB BSRAM
act_rptr  = Signal(16)                              # read address counter

# Param store
param_mem = Memory(shape=32, depth=512, init=[])   # 2 KiB BSRAM
```

---

## Prefetch Counter Logic

```python
# consume signal: asserted when mac4_next fires (funct3=0, funct7=1, start=1)
consume = Signal()

# Filter read pointer — cyclic wrap at k_count
with m.If(consume):
    with m.If(filt_rptr == k_count - 1):
        m.d.sync += filt_rptr.eq(0)
    with m.Else():
        m.d.sync += filt_rptr.eq(filt_rptr + 1)

# Activation read pointer — sequential, no wrap (firmware resets between channels)
with m.If(consume):
    m.d.sync += act_rptr.eq(act_rptr + 1)

# first/last from filter counter position
m.d.comb += first.eq(filt_rptr == 0)
m.d.comb += last.eq(filt_rptr == k_count - 1)

# Wire counters to SRAM read address ports
m.d.comb += filt_mem_rd.addr.eq(filt_rptr)
m.d.comb += act_mem_rd.addr.eq(act_rptr)
```

---

## Operand Mux (in engine.py top-level wiring)

MAC4 is unchanged. The top-level muxes the operand source based on funct7:

```python
is_mac4_next = Signal()
m.d.comb += is_mac4_next.eq(mac.start & mac.funct7[0])

# SRAM read data → MAC inputs when funct7=1, else CPU in0/in1
m.d.comb += [
    mac.in0.eq(Mux(is_mac4_next, act_mem_rd.data, self.cmd_in0)),
    mac.in1.eq(Mux(is_mac4_next, filt_mem_rd.data, self.cmd_in1)),
]

# Auto-increment on mac4_next
m.d.comb += consume.eq(is_mac4_next)

# first signal resets accumulator (OR'd with explicit reset)
m.d.comb += mac.reset_acc.eq(write.reset_acc | (first & is_mac4_next))
```

---

## WriteRegs Extension (funct7 sub-selection)

The existing `WriteRegs` (funct3=2) switches on `funct7` instead of `in0`:

```python
class WriteRegs(Instruction):
    def __init__(self):
        super().__init__()
        self.input_offset    = Signal(signed(32))
        self.reset_acc       = Signal()
        self.k_count         = Signal(16)
        self.filt_w_en       = Signal()
        self.act_w_en        = Signal()
        self.param_w_en      = Signal()
        self.filt_rptr_reset = Signal()
        self.act_rptr_reset  = Signal()

    def elaborate(self, platform):
        m = super().elaborate(platform)
        with m.If(self.start):
            with m.Switch(self.funct7):
                with m.Case(0): m.d.sync += self.input_offset.eq(self.in1)
                with m.Case(1): m.d.comb += self.reset_acc.eq(1)
                with m.Case(2): m.d.sync += self.k_count.eq(self.in1[:16])
                with m.Case(3): m.d.comb += self.filt_w_en.eq(1)
                with m.Case(4): m.d.comb += self.act_w_en.eq(1)
                with m.Case(5): m.d.comb += self.param_w_en.eq(1)
                with m.Case(6): m.d.comb += self.filt_rptr_reset.eq(1)
                with m.Case(7): m.d.comb += self.act_rptr_reset.eq(1)
            m.d.comb += self.done.eq(1)
        return m
```

For SRAM writes (funct7=3,4,5): `in0` = address, `in1` = data.
Wired in engine.py:

```python
m.d.comb += [
    filt_mem_wr.addr.eq(self.cmd_in0[:16]),
    filt_mem_wr.data.eq(self.cmd_in1),
    filt_mem_wr.en.eq(write.filt_w_en),
    # same for act_mem, param_mem
]
```

---

## Timing Invariant

```
  RULE 1: Prime before first mac4_next

    cfu.resetFilterRPtr()    // funct7=6: filt_rptr ← 0, sets SRAM read addr
    // ... at least 1 cycle gap (resetAcc, resetActRPtr, etc. provide this) ...
    cfu.mac4Next()           // SRAM data[0] is now on the output — valid ✓

  RULE 2: Write all data before prime

    for (0..len) |i| cfu.writeFilter(i, words[i]);   // write phase
    cfu.resetFilterRPtr();                             // prime — starts read phase

  RULE 3: Reset act read pointer between output channels

    for (0..n_channels) |n| {
        cfu.resetActRPtr();   // rewind to same spatial position
        // ... mac4_next loop (filter pointer continues advancing) ...
    }
```

---

## Comparison to CUDA Shared Memory

| Concept | CUDA Shared Memory | Your BSRAM + Prefetch |
|---|---|---|
| Declaration | `__shared__ float smem[SIZE]` | `Memory(shape=32, depth=512)` |
| Write | `smem[idx] = val` (thread writes) | `cfu.writeFilter(addr, data)` (CPU writes) |
| Read | `val = smem[idx]` (thread reads) | SRAM output auto-presented by prefetch counter |
| Bank conflicts | 32 banks, conflicts stall | Single-port per store, no conflicts |
| Address generation | Explicit in kernel code | Hardware counter with cyclic wrap |
| Latency hiding | Warp scheduling | Prefetch: addr set 1 cycle before data needed |
