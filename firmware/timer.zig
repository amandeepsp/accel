const mmio = @import("mmio.zig");
const csr = @import("csr");

const Load = mmio.Reg(csr.timer0_load);
const Reload = mmio.Reg(csr.timer0_reload);
const Enable = mmio.Reg(csr.timer0_en);
const UpdateValue = mmio.Reg(csr.timer0_update_value);
const Value = mmio.Reg(csr.timer0_value);

pub fn init() void {
    // Configure LiteX timer0 as a free-running down-counter.
    Load.write(0xFFFF_FFFF);
    Reload.write(0xFFFF_FFFF);
    Enable.write(1);
}

/// Read the low 16 bits of the elapsed cycle counter.
/// End-to-end deltas taken with this helper include this CSR/MMIO sampling
/// overhead at both the start and end of the measured region.
pub inline fn readCycles() u16 {
    UpdateValue.write(1);
    return @truncate(0xFFFF_FFFF -% Value.read());
}
