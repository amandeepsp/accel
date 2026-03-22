# CPU-as-Sequencer with K-Unrolled Firmware Loop

## Architecture Overview

The VexRiscv CPU IS the sequencer. The firmware runs the triple-nested loop
(spatial × N × K/4) directly. The only new hardware is:

- Two `Memory` instances (filter, activation) — synthesize to BSRAM
- Two read-pointer counters + cyclic wrap logic (~15 lines Amaranth)
- One operand mux per MAC input (funct7=0 → CPU regs, funct7=1 → SRAM output)

The existing `SimdMac4`, `SRDHM`, and `RoundingDividebyPOT` are unchanged.

---

## CFU Instruction ABI

All instructions use the existing R-type CFU encoding. `funct3` selects the
instruction (0–7), `funct7` sub-selects the variant or register.

```
funct3  funct7  Name                  in0         in1          output    Cycles
──────  ──────  ────                  ───         ───          ──────    ──────
  0       0     MAC4                  activations weights      (acc)     1
  0       1     MAC4_NEXT             (ignored)   (ignored)    (acc)     1
                Reads SRAM[filt_rptr] and SRAM[act_rptr] → MAC4.
                Auto-increments both pointers. Filter wraps at k_count.

  1       0     READ input_offset                              value     1
  1       1     READ accumulator                               value     1

  2       0     WRITE input_offset                in1=value              1
  2       1     RESET accumulator                                        1
  2       2     WRITE k_count                     in1=value              1
  2       3     WRITE filter SRAM     in0=addr    in1=data               1
  2       4     WRITE act SRAM        in0=addr    in1=data               1
  2       5     WRITE param SRAM      in0=addr    in1=data               1
  2       6     RESET filter read ptr (prime)                            1
  2       7     RESET act read ptr                                       1

  3       0     SRDHM                 a           b            result    2
  4       0     RDBPOT                x           shift        result    1
  5-7           (free for future use)
```

---

## Firmware Compute Loop

```zig
const cfu = @import("cfu.zig");

const LayerConfig = struct {
    k_count: u16,       // K / 4
    n_channels: u16,    // N
    spatial: u16,       // S = H * W
    input_offset: i8,
    output_offset: i8,
    act_min: i8,
    act_max: i8,
};

fn computeLayer(cfg: LayerConfig) void {
    cfu.setInputOffset(cfg.input_offset);
    cfu.setKCount(cfg.k_count);

    for (0..cfg.spatial) |s| {
        _ = s;
        cfu.resetFilterRPtr();   // prime: SRAM addr ← 0

        for (0..cfg.n_channels) |n| {
            cfu.resetAcc();
            cfu.resetActRPtr();  // rewind activation read to start of spatial pos

            // K-unrolled inner loop
            mac4Burst(cfg.k_count);

            // Requant (CPU-driven, uses existing SRDHM + RDBPOT instructions)
            const acc = cfu.readAcc();
            const bias = readParamBias(n);
            const mult = readParamMult(n);
            const shift = readParamShift(n);

            var result = cfu.srdhm(acc + bias, mult);
            result = cfu.rdbpot(result, shift);
            result = result + cfg.output_offset;
            result = clamp(result, cfg.act_min, cfg.act_max);

            uart.sendI8(@truncate(result));
        }
    }
}

fn mac4Burst(k_count: u16) void {
    var k: u16 = 0;
    while (k + 4 <= k_count) : (k += 4) {
        _ = cfu.mac4Next();
        _ = cfu.mac4Next();
        _ = cfu.mac4Next();
        _ = cfu.mac4Next();
    }
    while (k < k_count) : (k += 1) {
        _ = cfu.mac4Next();
    }
}

inline fn clamp(val: i32, min_val: i8, max_val: i8) i8 {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return @truncate(val);
}
```

---

## BSRAM Data Layout Contract

The firmware must write data into BSRAM in the order the prefetch counters
will read it:

**Filter BSRAM layout (row-major by output channel):**

```
addr:   0         1        ...   K/4-1     K/4       K/4+1    ...   2*K/4-1
data:  ch0_w[0]  ch0_w[1] ...  ch0_w[K/4-1]  ch1_w[0]  ch1_w[1] ...  ch1_w[K/4-1]
       ├────── channel 0 ──────┤├────── channel 1 ──────────────────┤
```

The filter read pointer wraps at `k_count` — so it reads ch0's weights,
wraps back to 0... wait, that's wrong. The filter pointer does NOT wrap
back to 0 between channels. It wraps cyclically every `k_count` entries
relative to its start position. Let's trace:

```
Channel 0: filt_rptr reads 0, 1, ..., k_count-1, wraps to 0
Channel 1: filt_rptr reads 0, 1, ..., k_count-1, wraps to 0
```

Wait — the filter pointer wraps to 0 every k_count. So channel 1 reads
the SAME filter data as channel 0? No — we need different weights per
channel.

**Solution:** The filter store is laid out with ALL channels' weights for
a given K index interleaved, OR the firmware reloads filter data between
channels.

**Correct approach for CPU-as-sequencer:** The filter store holds weights
for ONE output channel at a time (k_count entries). Between channels, the
firmware doesn't need to reload — it just lets the pointer wrap, and the
SAME k_count entries are re-read. But those entries need to be DIFFERENT
weights for each channel.

Two options:

1. **Full filter preload:** Store all N×K/4 entries. The read pointer runs
   sequentially without wrapping. Set k_count = N × K/4 (no cyclic wrap).
   The firmware tracks first/last in software instead. Simple but uses more
   BSRAM.

2. **Per-channel reload:** Load one channel's weights, compute, load next
   channel's weights, compute. Uses only K/4 entries of BSRAM but slower
   (reload overhead per channel).

3. **Sequential read with SW first/last:** Store all N channels' weights
   sequentially. Read pointer just increments (never wraps). Firmware
   knows k_count and manages accumulator reset + requant at the right
   boundaries. The hardware `first`/`last` signals are unused — firmware
   handles this logic.

**Option 3 is simplest for CPU-as-sequencer.** The filter read pointer is
just a sequential counter. The firmware explicitly resets the accumulator
and triggers requant — it already does this in the loop structure.

Revised filter layout:

```
addr:  0          1         ...  K/4-1      K/4        ...  2*K/4-1    ...
data:  ch0_w[0]   ch0_w[1]  ... ch0_w[-1]  ch1_w[0]   ... ch1_w[-1]  ...
       ├── channel 0 K/4 ──┤├── channel 1 K/4 ──────┤
```

The read pointer just increments on every mac4_next. No wrap needed.
The firmware loop structure provides the channel boundaries.

**Activation BSRAM layout (one spatial position's K bytes):**

```
addr:  0          1         ...  K/4-1
data:  act[0:3]   act[4:7]  ... act[K-4:K-1]
       ├── K bytes for spatial position s ──┤
```

The activation read pointer resets to 0 at the start of each output channel
(same activations, different weights).

---

## Utilization Analysis

For K=8 (k_count=2), N=4, S=1:

```
Per output channel:
  resetAcc       1 cycle
  resetActRPtr   1 cycle
  mac4Next × 2   2 cycles  ← MAC active
  readAcc        1 cycle
  requant        ~5 cycles (srdhm=2 + rdbpot=1 + clamp+store=2)
  Total:         ~10 cycles, 2 MAC active

4 channels × 10 = 40 cycles total
MAC active: 8 / 40 = 20%
```

For K=32 (k_count=8), N=32, S=16:

```
Per output channel:
  overhead:  2 cycles
  mac4Next: 8 cycles (unrolled, no loop overhead)
  requant:  5 cycles
  Total:    15 cycles, 8 MAC active

512 elements × 15 = 7680 cycles
MAC active: 4096 / 7680 = 53%
```

The requant overhead dominates for small K. The HW FSM sequencer
(stretch goal) pipelines requant with the next channel's MACs, hiding
this overhead. For the CPU-driven approach, this is the main cost.

---

## Comparison to HW FSM Sequencer

| Aspect | CPU-as-Sequencer | HW FSM Sequencer |
|---|---|---|
| Hardware complexity | ~15 lines (counters + mux) | ~100 lines (FSM + counters + PP pipeline) |
| New Amaranth modules | 0 (wiring in engine.py) | Sequencer FSM, PostProcess pipeline |
| MAC utilization (K=8) | 20-67% (depends on requant) | 80% |
| MAC utilization (K=32) | 53-89% | 94% |
| Debugging | printf in firmware loop | VCD waveform analysis |
| Requant overhead | Visible (CPU cycles) | Hidden (pipelined) |
| Upgrade path | Drop-in replacement | — |

The CPU-as-sequencer is the right first step. Build the HW FSM when you
need the last 20-30% of utilization.
