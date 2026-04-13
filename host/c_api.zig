const std = @import("std");
const driver = @import("driver");

pub const AccelStatus = enum(c_int) {
    ok = 0,
    invalid_argument = 1,
    payload_too_large = 2,
    bad_response = 3,
    bad_magic = 4,
    unknown_op = 5,
    bad_payload_len = 6,
    bad_address = 7,
    illegal_instruction = 8,
    trap_fault = 9,
    device_error = 10,
    out_of_memory = 11,
    io_error = 12,

    pub fn describe(self: AccelStatus) [*:0]const u8 {
        return switch (self) {
            .ok => "ok",
            .invalid_argument => "invalid argument",
            .payload_too_large => "payload too large",
            .bad_response => "bad response",
            .bad_magic => "bad magic",
            .unknown_op => "unknown op",
            .bad_payload_len => "bad payload length",
            .bad_address => "bad address",
            .illegal_instruction => "illegal instruction",
            .trap_fault => "trap fault",
            .device_error => "device error",
            .out_of_memory => "out of memory",
            .io_error => "io error",
        };
    }

    pub fn fromError(err: anyerror) AccelStatus {
        return switch (err) {
            error.InvalidDimensions => .invalid_argument,
            error.PayloadTooLarge => .payload_too_large,
            error.BadResponse => .bad_response,
            error.BadMagic => .bad_magic,
            error.UnknownOp => .unknown_op,
            error.BadPayloadLen => .bad_payload_len,
            error.BadAddress => .bad_address,
            error.IllegalInstruction => .illegal_instruction,
            error.TrapFault => .trap_fault,
            error.DeviceError => .device_error,
            error.OutOfMemory => .out_of_memory,
            else => .io_error,
        };
    }
};

inline fn require(comptime T: type, opt: ?T) union(enum) { ok: T, err: c_int } {
    if (opt) |v| return .{ .ok = v };
    return .{ .err = @intFromEnum(AccelStatus.invalid_argument) };
}

pub const AccelHandle = struct {
    driver: driver.Driver,
};

pub export fn accel_open(
    port_path: ?[*:0]const u8,
    baud_rate: u32,
    out_handle: ?*?*AccelHandle,
) c_int {
    const path = switch (require([*:0]const u8, port_path)) {
        .ok => |v| v,
        .err => |e| return e,
    };
    const handle_ptr = switch (require(*?*AccelHandle, out_handle)) {
        .ok => |v| v,
        .err => |e| return e,
    };

    const handle = std.heap.page_allocator.create(AccelHandle) catch {
        handle_ptr.* = null;
        return @intFromEnum(AccelStatus.out_of_memory);
    };
    errdefer std.heap.page_allocator.destroy(handle);

    handle.driver = driver.Driver.init(std.mem.span(path), baud_rate) catch |err| {
        handle_ptr.* = null;
        return @intFromEnum(AccelStatus.fromError(err));
    };

    handle_ptr.* = handle;
    return @intFromEnum(AccelStatus.ok);
}

pub export fn accel_close(handle: ?*AccelHandle) void {
    if (handle) |h| {
        h.driver.deinit();
        std.heap.page_allocator.destroy(h);
    }
}

pub export fn accel_ping(handle: ?*AccelHandle) c_int {
    const h = switch (require(*AccelHandle, handle)) {
        .ok => |v| v,
        .err => |e| return e,
    };
    h.driver.ping() catch |err| return @intFromEnum(AccelStatus.fromError(err));
    return @intFromEnum(AccelStatus.ok);
}

pub export fn accel_last_cycles(handle: ?*AccelHandle) u16 {
    const h = handle orelse return 0;
    return h.driver.last_cycles;
}

pub export fn accel_write_mem(
    handle: ?*AccelHandle,
    addr: u32,
    data: ?[*]const u8,
    len: usize,
) c_int {
    const h = switch (require(*AccelHandle, handle)) {
        .ok => |v| v,
        .err => |e| return e,
    };
    const ptr = switch (require([*]const u8, data)) {
        .ok => |v| v,
        .err => |e| return e,
    };

    h.driver.writeMem(addr, ptr[0..len]) catch |err| {
        return @intFromEnum(AccelStatus.fromError(err));
    };
    return @intFromEnum(AccelStatus.ok);
}

pub export fn accel_read_mem(
    handle: ?*AccelHandle,
    addr: u32,
    buf: ?[*]u8,
    len: usize,
) c_int {
    const h = switch (require(*AccelHandle, handle)) {
        .ok => |v| v,
        .err => |e| return e,
    };
    const ptr = switch (require([*]u8, buf)) {
        .ok => |v| v,
        .err => |e| return e,
    };

    h.driver.readMem(addr, ptr[0..len]) catch |err| {
        return @intFromEnum(AccelStatus.fromError(err));
    };
    return @intFromEnum(AccelStatus.ok);
}

pub export fn accel_exec(
    handle: ?*AccelHandle,
    program: ?[*]const u8,
    program_len: usize,
    out_cycles: ?*u32,
) c_int {
    const h = switch (require(*AccelHandle, handle)) {
        .ok => |v| v,
        .err => |e| return e,
    };
    const ptr = switch (require([*]const u8, program)) {
        .ok => |v| v,
        .err => |e| return e,
    };
    const cycles_ptr = switch (require(*u32, out_cycles)) {
        .ok => |v| v,
        .err => |e| return e,
    };

    cycles_ptr.* = h.driver.exec(ptr[0..program_len]) catch |err| {
        return @intFromEnum(AccelStatus.fromError(err));
    };
    return @intFromEnum(AccelStatus.ok);
}

pub export fn accel_status_string(code: c_int) [*:0]const u8 {
    if (std.meta.intToEnum(AccelStatus, code)) |status| {
        return status.describe();
    } else |_| {
        return "unknown status";
    }
}
