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

    // Test mac4: [1,2,3,4] · [5,6,7,8] = 1*5 + 2*6 + 3*7 + 4*8 = 70
    const mac_result = try drv.mac4(.{ 1, 2, 3, 4 }, .{ 5, 6, 7, 8 });
    std.debug.print("mac4 result = {} (expected 70)\n", .{mac_result});

    // Test srdhm: (1000000 * 1000000 + (1<<30)) >> 31
    // = (1e12 + 1073741824) >> 31 = 1001073741824 >> 31 = 466
    const srdhm_result = try drv.srdhm(1000000, 1000000);
    std.debug.print("srdhm result = {} (expected 466)\n", .{srdhm_result});

    // Test rdbpot: 100 >> 2 = 25
    const rdbpot_result = try drv.rdbpot(100, 2);
    std.debug.print("rdbpot result = {} (expected 25)\n", .{rdbpot_result});
}
