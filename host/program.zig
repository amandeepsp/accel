const std = @import("std");
const ir = @import("ir");

pub const Program = struct {
    alloc: std.mem.Allocator,
    tensors: std.ArrayListUnmanaged(ir.TensorDescriptor) = .{},
    code: std.ArrayListUnmanaged(u8) = .{},
    instruction_count: u16 = 0,

    pub fn init(alloc: std.mem.Allocator) !Program {
        return Program{
            .alloc = alloc,
        };
    }

    pub fn deinit(self: *Program) void {
        self.tensors.deinit(self.alloc);
        self.code.deinit(self.alloc);
    }

    pub fn addTensor(
        self: *Program,
        addr: u32,
        dims: [2]u16,
        stride_row: u16,
        dtype: ir.TensorDtype,
    ) !u8 {
        if (self.tensors.items.len >= std.math.maxInt(u8)) {
            return error.TooManyTensors;
        }

        const tensor_id: u8 = @truncate(self.tensors.items.len);
        try self.tensors.append(self.alloc, .{
            .base_addr = addr,
            .dim0 = dims[0],
            .dim1 = dims[1],
            .stride_row = stride_row,
            .dtype = dtype,
            .flags = 0x0,
            ._padding = 0x0,
        });

        return tensor_id;
    }

    pub fn emitInstruction(self: *Program, instruction: anytype) !void {
        comptime {
            const T = @TypeOf(instruction);
            const type_info = @typeInfo(T);
            if (type_info != .@"struct" or type_info.@"struct".layout != .@"packed") {
                @compileError("Instruction must be packed");
            }

            if (!@hasField(T, "opcode")) {
                @compileError("Missing opcode");
            }
        }
        try self.code.appendSlice(self.alloc, std.mem.asBytes(&instruction));
        self.instruction_count += 1;
    }
};
