const std = @import("std");
const driver = @import("driver");

pub fn main() !void {
    var dbg_alloc = std.heap.DebugAllocator(.{}){};
    defer _ = dbg_alloc.deinit();

    var args = std.process.args();
    _ = args.next(); // skip argv[0]
    const port = args.next() orelse "/dev/ttyUSB1";
    const command = args.next() orelse "ping";

    var drv = try driver.Driver.init(port, 115200);
    defer drv.deinit();

    if (std.mem.eql(u8, command, "ping")) {
        try drv.ping();
        std.debug.print("ping ok, last_cycles={}\n", .{drv.last_cycles});
        return;
    }

    if (std.mem.eql(u8, command, "exec-bin")) {
        const path = args.next() orelse return error.InvalidArgument;
        const program = try readFileAlloc(dbg_alloc.allocator(), path);
        defer dbg_alloc.allocator().free(program);

        const cycles = try drv.exec(program);
        std.debug.print("exec-bin ok, cycles={}, last_cycles={}\n", .{ cycles, drv.last_cycles });
        return;
    }

    if (std.mem.eql(u8, command, "write-file")) {
        const addr = try parseAddr(args.next() orelse return error.InvalidArgument);
        const path = args.next() orelse return error.InvalidArgument;
        const data = try readFileAlloc(dbg_alloc.allocator(), path);
        defer dbg_alloc.allocator().free(data);

        try drv.writeMem(addr, data);
        std.debug.print("write-file ok, addr=0x{x}, bytes={}\n", .{ addr, data.len });
        return;
    }

    if (std.mem.eql(u8, command, "read-file")) {
        const addr = try parseAddr(args.next() orelse return error.InvalidArgument);
        const len = try parseLen(args.next() orelse return error.InvalidArgument);
        const path = args.next() orelse return error.InvalidArgument;

        const buf = try dbg_alloc.allocator().alloc(u8, len);
        defer dbg_alloc.allocator().free(buf);

        try drv.readMem(addr, buf);
        try std.fs.cwd().writeFile(.{ .sub_path = path, .data = buf });
        std.debug.print("read-file ok, addr=0x{x}, bytes={}, path={s}\n", .{ addr, len, path });
        return;
    }

    std.debug.print("unknown command: {s}\n", .{command});
    return error.InvalidArgument;
}

fn parseAddr(text: []const u8) !u32 {
    return try std.fmt.parseInt(u32, text, 0);
}

fn parseLen(text: []const u8) !usize {
    return try std.fmt.parseInt(usize, text, 0);
}

fn readFileAlloc(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return try std.fs.cwd().readFileAlloc(allocator, path, std.math.maxInt(usize));
}
