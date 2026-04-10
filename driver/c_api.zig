const std = @import("std");
const driver = @import("driver");

pub const AccelStatus = enum(c_int) {
    ok = 0,
    invalid_argument = 1,
    invalid_dimensions = 2,
    payload_too_large = 3,
    bad_response = 4,
    bad_magic = 5,
    unknown_op = 6,
    bad_payload_len = 7,
    illegal_instruction = 8,
    trap_fault = 9,
    device_error = 10,
    out_of_memory = 11,
    io_error = 12,

    pub fn describe(self: AccelStatus) [*:0]const u8 {
        return switch (self) {
            .ok => "ok",
            .invalid_argument => "invalid argument",
            .invalid_dimensions => "invalid dimensions",
            .payload_too_large => "payload too large",
            .bad_response => "bad response",
            .bad_magic => "bad magic",
            .unknown_op => "unknown op",
            .bad_payload_len => "bad payload length",
            .illegal_instruction => "illegal instruction",
            .trap_fault => "trap fault",
            .device_error => "device error",
            .out_of_memory => "out of memory",
            .io_error => "io error",
        };
    }

    pub fn fromError(err: anyerror) AccelStatus {
        return switch (err) {
            error.InvalidDimensions => .invalid_dimensions,
            error.PayloadTooLarge => .payload_too_large,
            error.BadResponse => .bad_response,
            error.BadMagic => .bad_magic,
            error.UnknownOp => .unknown_op,
            error.BadPayloadLen => .bad_payload_len,
            error.IllegalInstruction => .illegal_instruction,
            error.TrapFault => .trap_fault,
            error.DeviceError => .device_error,
            error.OutOfMemory => .out_of_memory,
            else => .io_error,
        };
    }
};

/// Unwrap an optional or return invalid_argument.
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

pub export fn accel_mma(
    handle: ?*AccelHandle,
    accum_ptr: ?[*]i32,
    accum_len: usize,
    lhs_ptr: ?[*]const i8,
    lhs_len: usize,
    rhs_ptr: ?[*]const i8,
    rhs_len: usize,
    m: usize,
    n: usize,
    k: usize,
) c_int {
    const h = switch (require(*AccelHandle, handle)) {
        .ok => |v| v,
        .err => |e| return e,
    };
    const accum = switch (require([*]i32, accum_ptr)) {
        .ok => |v| v,
        .err => |e| return e,
    };
    const lhs = switch (require([*]const i8, lhs_ptr)) {
        .ok => |v| v,
        .err => |e| return e,
    };
    const rhs = switch (require([*]const i8, rhs_ptr)) {
        .ok => |v| v,
        .err => |e| return e,
    };

    h.driver.mma(
        accum[0..accum_len],
        lhs[0..lhs_len],
        rhs[0..rhs_len],
        m,
        n,
        k,
    ) catch |err| return @intFromEnum(AccelStatus.fromError(err));

    return @intFromEnum(AccelStatus.ok);
}

pub export fn accel_status_string(code: c_int) [*:0]const u8 {
    if (std.meta.intToEnum(AccelStatus, code)) |status| {
        return status.describe();
    } else |_| {
        return "unknown status";
    }
}
