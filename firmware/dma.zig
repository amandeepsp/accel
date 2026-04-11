const csr = @import("csr");
const mmio = @import("mmio.zig");

pub fn Dma(
    comptime base_addr: usize,
    comptime length_addr: usize,
    comptime enable_addr: usize,
    comptime done_addr: usize,
    comptime loop_addr: usize,
    comptime offset_addr: usize,
) type {
    return struct {
        const Self = @This();

        base: mmio.Reg(base_addr),
        length: mmio.Reg(length_addr),
        enable: mmio.Reg(enable_addr),
        done: mmio.Reg(done_addr),
        loop: mmio.Reg(loop_addr),
        offset: mmio.Reg(offset_addr),

        pub fn init() Self {
            return .{
                .base = .{},
                .length = .{},
                .enable = .{},
                .done = .{},
                .loop = .{},
                .offset = .{},
            };
        }

        pub inline fn configure(self: Self, base_ptr: u32, len: u32) void {
            @TypeOf(self.base).write(base_ptr);
            @TypeOf(self.length).write(len);
        }

        pub inline fn setLoop(self: Self, enabled: bool) void {
            @TypeOf(self.loop).write(@intFromBool(enabled));
        }

        pub inline fn start(self: Self) void {
            @TypeOf(self.enable).write(1);
        }

        pub inline fn kick(self: Self, base_ptr: u32, len: u32) void {
            self.stop();
            self.setLoop(false);
            self.configure(base_ptr, len);
            self.start();
        }

        pub inline fn stop(self: Self) void {
            @TypeOf(self.enable).write(0);
        }

        pub inline fn isDone(self: Self) bool {
            return @TypeOf(self.done).read() != 0;
        }

        pub inline fn wait(self: Self) void {
            while (!self.isDone()) {}
        }

        pub inline fn offsetBytes(self: Self) u32 {
            return @TypeOf(self.offset).read();
        }
    };
}

pub const Act = Dma(
    csr.act_dma_base,
    csr.act_dma_length,
    csr.act_dma_enable,
    csr.act_dma_done,
    csr.act_dma_loop,
    csr.act_dma_offset,
);

pub const Wgt = Dma(
    csr.wgt_dma_base,
    csr.wgt_dma_length,
    csr.wgt_dma_enable,
    csr.wgt_dma_done,
    csr.wgt_dma_loop,
    csr.wgt_dma_offset,
);
