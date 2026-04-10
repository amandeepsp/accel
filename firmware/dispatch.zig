const link = @import("link.zig");
const memory = @import("memory.zig");

pub fn dispatch(header: link.Header) void {
    switch (header.op) {
        .ping => ping(header),
        .read => memory.readMem(header),
        .write => memory.writeMem(header),
        else => link.sendError(header.seq_id, .unknown_op),
    }
}

fn ping(header: link.Header) void {
    if (header.payload_len != 0) {
        link.drainPayload(header.payload_len);
        link.sendError(header.seq_id, .bad_payload_len);
        return;
    }

    link.sendOk(header.seq_id, &.{}, 0);
}
