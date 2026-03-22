# Optional Public Artifacts

This file is intentionally separate from the main course text.

The course itself should stay focused on ML accelerator concepts, implementation details, and design tradeoffs. If you want to keep a public log, progress thread, or build diary, use lightweight artifacts like these.

## Good Artifact Types

- one system diagram
- one latency or utilization table
- one waveform screenshot with a short explanation
- one memory map or tile-size sketch
- one tradeoff note: what you chose and why
- one postmortem: what broke and what changed your mind

## Simple Template

Use a short structure:

1. what I was trying to learn
2. what I built or measured
3. what surprised me
4. what I would change next

## Unit-Sized Ideas

- `00-architecture.md` - host/bus/device diagram
- `01-compute.md` - decoded custom instruction and MAC datapath sketch
- `02-datapath.md` - launch/transfer/compute breakdown
- `03-fusion.md` - unfused vs fused dataflow diagram
- `04-engine.md` - control-model comparison table
- `05-compiler.md` - one-op lowering trace
- `06-model.md` - operator support matrix
- `07-systolic.md` - area vs throughput comparison
- `08-feeding.md` - queue / DMA / overlap timeline
- `09-modern.md` - workload mismatch table
- `10-redesign.md` - v2 architecture memo

Keep the artifact small enough that someone else can understand it quickly and challenge the reasoning behind it.
