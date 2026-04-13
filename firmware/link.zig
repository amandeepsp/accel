const uart = @import("uart.zig");
const std = @import("std");
const protocol = @import("protocol");

pub const MAGIC_REQ = protocol.MAGIC_REQ;
pub const MAGIC_RESP = protocol.MAGIC_RESP;
pub const OpType = protocol.OpType;
pub const StatusCode = protocol.StatusCode;
pub const Header = protocol.RequestHeader;

pub const LinkError = error{ BadMagic, Timeout };

pub fn recvHeader() LinkError!protocol.RequestHeader {
    // Sync: skip bytes until we find the magic request byte.
    while (true) {
        const b = uart.readByte();
        if (b == MAGIC_REQ) break;
    }
    // Read remaining 7 bytes of the header.
    var header: protocol.RequestHeader = undefined;
    const buf: []u8 = std.mem.asBytes(&header);
    buf[0] = MAGIC_REQ;
    uart.readBytes(buf[1..]);
    return header;
}

pub fn sendResponse(seq_id: u16, status: StatusCode, data: []const u8, cycles: u16) void {
    sendResponseHeader(seq_id, status, @intCast(data.len), cycles);
    uart.writeBytes(data);
}

pub fn sendResponseHeader(seq_id: u16, status: StatusCode, payload_len: u16, cycles: u16) void {
    const rsp = protocol.ResponseHeader{
        .status = status,
        .payload_len = payload_len,
        .seq_id = seq_id,
        .cycles_lo = cycles,
    };
    uart.writeBytes(std.mem.asBytes(&rsp));
}

pub fn sendOk(seq_id: u16, data: []const u8, cycles: u16) void {
    sendResponse(seq_id, .ok, data, cycles);
}

pub fn sendError(seq_id: u16, code: StatusCode, debug_data: []const u8) void {
    sendResponse(seq_id, code, debug_data, 0);
}

pub fn sendDebug(value: u32) void {
    uart.writeByte('D');
    uart.writeByte('E');
    uart.writeByte('B');
    uart.writeByte('G');
    uart.writeBytes(std.mem.asBytes(&value));
}

pub fn drainPayload(len: usize) void {
    var remaining = len;
    var buf: [64]u8 = undefined;
    while (remaining > 0) {
        const chunk_len = @min(remaining, buf.len);
        uart.readBytes(buf[0..chunk_len]);
        remaining -= chunk_len;
    }
}
