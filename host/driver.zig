const std = @import("std");
const serial = @import("serial");
const protocol = @import("protocol");

const log = std.log.scoped(.accel);

pub const AccelError = error{
    PayloadTooLarge,
    BadResponse,
    BadMagic,
    UnknownOp,
    BadPayloadLen,
    BadAddress,
    IllegalInstruction,
    TrapFault,
    DeviceError,
};

pub const Driver = struct {
    port: std.fs.File,
    last_cycles: u16 = 0,

    pub fn init(port_path: []const u8, baud_rate: u32) !Driver {
        const port = try std.fs.openFileAbsolute(port_path, .{ .mode = .read_write });
        errdefer port.close();

        try serial.configureSerialPort(port, .{
            .baud_rate = baud_rate,
            .word_size = .eight,
        });
        try serial.flushSerialPort(port, .both);

        return .{ .port = port };
    }

    pub fn deinit(self: *Driver) void {
        self.port.close();
    }

    fn drainBytes(self: *Driver, len: usize) !void {
        var remaining = len;
        var buf: [256]u8 = undefined;
        while (remaining > 0) {
            const chunk_len = @min(remaining, buf.len);
            const read_len = try self.port.readAll(buf[0..chunk_len]);
            if (read_len != chunk_len) return error.BadResponse;
            remaining -= chunk_len;
        }
    }

    fn issuePayloadParts(
        self: *Driver,
        op: protocol.OpType,
        payload_a: []const u8,
        payload_b: []const u8,
        response: []u8,
    ) !void {
        const payload_len = payload_a.len + payload_b.len;
        if (payload_len > std.math.maxInt(u16)) return error.PayloadTooLarge;

        const header = protocol.RequestHeader.init(op, @intCast(payload_len), 0);
        try self.port.writeAll(header.as_bytes());
        try self.port.writeAll(payload_a);
        try self.port.writeAll(payload_b);

        var resp_header_buf: [@sizeOf(protocol.ResponseHeader)]u8 = undefined;
        const read_len = try self.port.readAll(&resp_header_buf);
        if (read_len != resp_header_buf.len) return error.BadResponse;

        const resp_header = protocol.ResponseHeader.from_bytes(&resp_header_buf);
        if (resp_header.magic != protocol.MAGIC_RESP) return error.BadMagic;

        if (!resp_header.status.isOk()) {
            log.err("device returned error: {s} (0x{X:0>2})", .{
                resp_header.status.describe(),
                @intFromEnum(resp_header.status),
            });
            if (resp_header.payload_len > 0) {
                var debug_buf: [256]u8 = undefined;
                const debug_len = @min(resp_header.payload_len, debug_buf.len);
                const debug_read = try self.port.readAll(debug_buf[0..debug_len]);
                if (debug_read > 0) {
                    log.err("debug payload ({d} bytes): {any}", .{
                        debug_read,
                        debug_buf[0..debug_read],
                    });
                }
                // Drain remaining bytes
                if (resp_header.payload_len > debug_read) {
                    try self.drainBytes(resp_header.payload_len - debug_read);
                }
            }
            return statusToError(resp_header.status);
        }

        self.last_cycles = resp_header.cycles_lo;

        if (resp_header.payload_len != response.len) {
            if (resp_header.payload_len > 0) {
                try self.drainBytes(resp_header.payload_len);
            }
            return error.BadResponse;
        }

        if (response.len > 0) {
            const read_payload_len = try self.port.readAll(response);
            if (read_payload_len != response.len) return error.BadResponse;
        }
    }

    fn issuePayload(
        self: *Driver,
        op: protocol.OpType,
        payload: []const u8,
        response: []u8,
    ) !void {
        try self.issuePayloadParts(op, payload, &.{}, response);
    }

    pub fn ping(self: *Driver) !void {
        var response: [0]u8 = .{};
        try self.issuePayload(.ping, &.{}, response[0..]);
    }

    pub fn writeMem(self: *Driver, addr: u32, data: []const u8) !void {
        const chunk_max = std.math.maxInt(u16) - @sizeOf(protocol.WriteMem.ReqHeader);
        var offset: usize = 0;

        while (offset < data.len) {
            const chunk_len = @min(chunk_max, data.len - offset);
            const req = protocol.WriteMem.ReqHeader{
                .addr = addr + @as(u32, @intCast(offset)),
            };
            var response: [0]u8 = .{};
            try self.issuePayloadParts(
                .write,
                std.mem.asBytes(&req),
                data[offset .. offset + chunk_len],
                response[0..],
            );
            offset += chunk_len;
        }
    }

    pub fn readMem(self: *Driver, addr: u32, buf: []u8) !void {
        const chunk_max = std.math.maxInt(u16);
        var offset: usize = 0;

        while (offset < buf.len) {
            const chunk_len = @min(chunk_max, buf.len - offset);
            const req = protocol.ReadMem.Req{
                .addr = addr + @as(u32, @intCast(offset)),
                .len = @intCast(chunk_len),
            };
            try self.issuePayload(.read, std.mem.asBytes(&req), buf[offset .. offset + chunk_len]);
            offset += chunk_len;
        }
    }

    pub fn exec(self: *Driver, program: []const u8) !u32 {
        var response: [4]u8 = undefined;
        try self.issuePayload(.exec, program, &response);
        return std.mem.readInt(u32, &response, .little);
    }
};

fn statusToError(status: protocol.StatusCode) AccelError {
    return switch (status) {
        .ok => unreachable,
        .unknown_op => error.UnknownOp,
        .bad_payload_len => error.BadPayloadLen,
        .bad_address => error.BadAddress,
        .bad_magic => error.BadMagic,
        .timeout => error.DeviceError,
        .illegal_instruction => error.IllegalInstruction,
        .trap_fault => error.TrapFault,
    };
}
