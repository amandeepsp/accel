const std = @import("std");
const driver = @import("driver");

pub fn main() !void {
    var dgb_alloc = std.heap.DebugAllocator(.{}){};
    defer _ = dgb_alloc.deinit();

    var args = std.process.args();
    _ = args.next(); // skip argv[0]
    const port = args.next() orelse "/dev/ttyUSB1";

    var drv = try driver.Driver.init(port, 115200);
    defer drv.deinit();

    try drv.ping();
    std.debug.print("ping ok, last_cycles={}\n", .{drv.last_cycles});
}
