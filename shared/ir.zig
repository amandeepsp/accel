const std = @import("std");

pub const program_magic: u32 = 0x4B495200; // ("KIR\0")
pub const program_version: u8 = 1;

pub const ProgramHeader = extern struct {
    magic: u32 = program_magic,
    version: u8 = program_version,
    num_tensors: u8,
    num_instructions: u16,
};

pub const TensorDtype = enum(u8) {
    signed8 = 0x0,
    signedi32 = 0x1,
};

pub const TensorDescriptor = extern struct {
    base_addr: u32, //address in main ram, managed by compiler
    dim0: u16,
    dim1: u16,
    stride_row: u16, // bytes between rows
    dtype: TensorDtype,
    flags: u8, // reserved
    _padding: u32,
};

pub const InstructionType = enum(u8) {
    tile_load_act = 0x01,
    tile_load_wgt = 0x02,
    tile_mma = 0x03,
    tile_store = 0x04,
    set_epilogue = 0x05,
    done = 0x06,
};

/// Load a tile into hw scratchpad via DMA
/// addr = desc.base + m_offset * desc.stride_row + k_offset
pub const TileLoadAct = extern struct {
    opcode: InstructionType,
    tensor_id: u8,
    m_offset: u16,
    k_offset: u16,
    k_words: u16,
};

/// Load a weight tile, similar semantics to the activations.
pub const TileLoadWgt = extern struct {
    opcode: InstructionType,
    tensor_id: u8,
    n_offset: u16,
    k_offset: u16,
    k_words: u16,
};

/// Issue Compute, non-blocking, also waits for pending DMAs before issuing.
pub const TileMma = extern struct {
    opcode: InstructionType,
    flags: packed struct(u8) {
        first: bool,
        last: bool,
        _reserved: u6 = 0,
    },
    k_count: u16,
};

/// Store compute results to main RAM after pending compute completes.
pub const TileStore = extern struct {
    opcode: InstructionType,
    tensor_id: u8,
    m_offset: u16,
    n_offset: u16,
    m_count: u8,
    n_count: u8,
};

/// Load per-channel epilogue parameters and global activation config.
pub const SetEpilogue = extern struct {
    opcode: InstructionType,
    bias_tid: u8,
    mult_tid: u8,
    shift_tid: u8,
    n_offset: u16,
    n_count: u16,
    output_offset: i8,
    act_min: i8,
    act_max: i8,
    _padding: u8 = 0,
};

/// Explicit program terminator.
pub const Done = extern struct {
    opcode: InstructionType,
    _padding: [3]u8 = .{ 0, 0, 0 },
};
