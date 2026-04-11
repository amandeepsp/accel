const cpu_csr = @import("cpu_csr.zig");

/// Read the low 16 bits of the machine cycle counter.
/// End-to-end deltas taken with this helper include only CSR sampling overhead.
pub inline fn readCycles() u16 {
    return @truncate(cpu_csr.mcycle.read());
}
