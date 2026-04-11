const mmio = @import("mmio.zig");

/// VexRiscv CfuPlugin enable register (bit 31 enables CFU instructions).
pub const cfu_ctrl = mmio.Csr(0xBC0);

/// RISC-V machine trap state.
pub const mepc = mmio.Csr(0x341);
pub const mcause = mmio.Csr(0x342);
pub const mtval = mmio.Csr(0x343);

/// RISC-V machine cycle counter.
pub const mcycle = mmio.Csr(0xB00);
pub const mcycleh = mmio.Csr(0xB80);
