const link = @import("link.zig");
const interpreter = @import("interpreter.zig");
const memory = @import("memory.zig");
const std = @import("std");

pub fn dispatch(header: link.Header) void {
    switch (header.op) {
        .ping => ping(header),
        .read => memory.readMem(header),
        .write => memory.writeMem(header),
        .exec => exec(header),
        else => link.sendError(header.seq_id, .unknown_op, &.{}),
    }
}

fn ping(header: link.Header) void {
    if (header.payload_len != 0) {
        link.drainPayload(header.payload_len);
        link.sendError(header.seq_id, .bad_payload_len, &.{});
        return;
    }

    link.sendOk(header.seq_id, &.{}, 0);
}

fn exec(header: link.Header) void {
    var debug_buf: [8]u8 = undefined;
    const cycles = interpreter.execute(header.payload_len, &debug_buf) catch |err| {
        const code: link.StatusCode = switch (err) {
            error.BadMagic => .bad_magic,
            error.BadPayloadLen => .bad_payload_len,
            error.BadAddress => .bad_address,
        };
        link.sendError(header.seq_id, code, &debug_buf);
        return;
    };

    link.sendResponse(header.seq_id, .ok, std.mem.asBytes(&cycles), @truncate(cycles));
}
