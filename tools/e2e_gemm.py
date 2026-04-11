#!/usr/bin/env python3

import argparse
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_BIN = REPO_ROOT / "zig-out" / "bin" / "driver"
DEFAULT_CFU_WORD_BITS = int(os.environ.get("ACCEL_CFU_WORD_BITS", "64"))
DEFAULT_CFU_STORE_DEPTH_WORDS = int(
    os.environ.get("ACCEL_CFU_STORE_DEPTH_WORDS", "512")
)
DEFAULT_DRIVER_TIMEOUT_S = float(os.environ.get("ACCEL_DRIVER_TIMEOUT_S", "120"))
TENSOR_POOL_BASE = 0x40010000
MEM_ALIGN = 32

# IR Constants
KIR_MAGIC = 0x4B495200  # "KIR\0"
KIR_VERSION = 1

# Instruction opcodes
TILE_LOAD_ACT = 0x01
TILE_LOAD_WGT = 0x02
TILE_MMA = 0x03
TILE_STORE = 0x04
SET_EPILOGUE = 0x05
DONE = 0x06

# Tensor dtype
DTYPE_I8 = 0x0
DTYPE_I32 = 0x1


@dataclass(frozen=True)
class GemmShape:
    m: int
    k: int
    n: int
    tile: int

    def validate(self) -> None:
        if self.tile <= 0:
            raise ValueError("tile size must be positive")
        if self.m <= 0 or self.k <= 0 or self.n <= 0:
            raise ValueError("M, K, and N must be positive")
        if self.m != self.tile:
            raise ValueError(
                f"M tiling is not supported yet; M must match tile size {self.tile}, got {self.m}"
            )
        if self.n != self.tile:
            raise ValueError(
                f"N must match tile size {self.tile} for the current weight loader, got {self.n}"
            )


@dataclass(frozen=True)
class MemoryLayout:
    bias_addr: int
    mult_addr: int
    shift_addr: int
    weights_addr: int
    input_addr: int
    output_addr: int


def run(cmd: list[str], *, timeout_s: float) -> str:
    """Run the Zig driver. Uses a timeout so a silent UART (no firmware / wrong port) cannot hang forever."""
    print(f"[e2e] {' '.join(cmd)}", flush=True)
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        if exc.stdout:
            sys.stdout.write(exc.stdout)
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        print(
            f"[e2e] driver subprocess timed out after {timeout_s}s: {' '.join(cmd)}\n"
            "[e2e] check port, power, and that firmware is running (e.g. [link] ready on UART).",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(124) from exc
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.stdout + proc.stderr


def align_up(value: int, alignment: int = MEM_ALIGN) -> int:
    return (value + alignment - 1) & -alignment


def pack_i8(values: list[int]) -> bytes:
    return struct.pack("<" + "b" * len(values), *values)


def pack_i32(values: list[int]) -> bytes:
    return struct.pack("<" + "i" * len(values), *values)


def plan_memory(
    input_size: int, weight_size: int, output_size: int, n_channels: int
) -> MemoryLayout:
    cursor = TENSOR_POOL_BASE

    bias_addr = cursor
    cursor = align_up(cursor + n_channels * 4)

    mult_addr = cursor
    cursor = align_up(cursor + n_channels * 4)

    shift_addr = cursor
    cursor = align_up(cursor + n_channels * 4)

    weights_addr = cursor
    cursor = align_up(cursor + weight_size)

    input_addr = cursor
    cursor = align_up(cursor + input_size)

    output_addr = cursor
    _ = align_up(cursor + output_size)

    return MemoryLayout(
        bias_addr=bias_addr,
        mult_addr=mult_addr,
        shift_addr=shift_addr,
        weights_addr=weights_addr,
        input_addr=input_addr,
        output_addr=output_addr,
    )


class ProgramBuilder:
    """Build a Kernel IR program bytecode."""

    def __init__(self):
        self.tensors: list[
            tuple[int, int, int, int, int]
        ] = []  # (addr, dim0, dim1, stride, dtype)
        self.code = bytearray()
        self.code_offset = 0
        self.instruction_count = 0

    def add_tensor(
        self, addr: int, dim0: int, dim1: int, stride: int, dtype: int = DTYPE_I8
    ) -> int:
        """Add a tensor descriptor, return its ID."""
        tensor_id = len(self.tensors)
        self.tensors.append((addr, dim0, dim1, stride, dtype))
        return tensor_id

    def emit_u8(self, val: int) -> None:
        self.code.append(val & 0xFF)

    def emit_u16_le(self, val: int) -> None:
        self.code.extend(struct.pack("<H", val))

    def emit_u32_le(self, val: int) -> None:
        self.code.extend(struct.pack("<I", val))

    def emit_i8(self, val: int) -> None:
        self.code.append(val & 0xFF)

    def tile_load_act(
        self, tensor_id: int, m_offset: int, k_offset: int, k_words: int
    ) -> None:
        """Emit TILE_LOAD_ACT instruction."""
        self.emit_u8(TILE_LOAD_ACT)
        self.emit_u8(tensor_id)
        self.emit_u16_le(m_offset)
        self.emit_u16_le(k_offset)
        self.emit_u16_le(k_words)
        self.instruction_count += 1

    def tile_load_wgt(
        self, tensor_id: int, n_offset: int, k_offset: int, k_words: int
    ) -> None:
        """Emit TILE_LOAD_WGT instruction."""
        self.emit_u8(TILE_LOAD_WGT)
        self.emit_u8(tensor_id)
        self.emit_u16_le(n_offset)
        self.emit_u16_le(k_offset)
        self.emit_u16_le(k_words)
        self.instruction_count += 1

    def tile_mma(self, first: bool, last: bool, k_count: int) -> None:
        """Emit TILE_MMA (compute) instruction."""
        self.emit_u8(TILE_MMA)
        flags = (1 if first else 0) | (2 if last else 0)
        self.emit_u8(flags)
        self.emit_u16_le(k_count)
        self.instruction_count += 1

    def tile_store(
        self, tensor_id: int, m_offset: int, n_offset: int, m_count: int, n_count: int
    ) -> None:
        """Emit TILE_STORE instruction."""
        self.emit_u8(TILE_STORE)
        self.emit_u8(tensor_id)
        self.emit_u16_le(m_offset)
        self.emit_u16_le(n_offset)
        self.emit_u8(m_count)
        self.emit_u8(n_count)
        self.instruction_count += 1

    def set_epilogue(
        self,
        bias_id: int,
        mult_id: int,
        shift_id: int,
        n_offset: int,
        n_count: int,
        output_offset: int,
        act_min: int,
        act_max: int,
    ) -> None:
        """Emit SET_EPILOGUE instruction."""
        self.emit_u8(SET_EPILOGUE)
        self.emit_u8(bias_id)
        self.emit_u8(mult_id)
        self.emit_u8(shift_id)
        self.emit_u16_le(n_offset)
        self.emit_u16_le(n_count)
        self.emit_i8(output_offset)
        self.emit_i8(act_min)
        self.emit_i8(act_max)
        self.emit_u8(0)  # padding
        self.instruction_count += 1

    def done(self) -> None:
        """Emit DONE instruction."""
        self.emit_u8(DONE)
        self.emit_u8(0)
        self.emit_u8(0)
        self.emit_u8(0)
        self.instruction_count += 1

    def build(self) -> bytes:
        """Build the complete IR program with header and descriptors."""
        program = bytearray()

        # Header: magic, version, num_tensors, num_instructions
        program.extend(struct.pack("<I", KIR_MAGIC))
        program.append(KIR_VERSION)
        program.append(len(self.tensors))
        program.extend(struct.pack("<H", self.instruction_count))

        # Tensor descriptors
        for addr, dim0, dim1, stride, dtype in self.tensors:
            program.extend(struct.pack("<I", addr))
            program.extend(struct.pack("<HH", dim0, dim1))
            program.extend(struct.pack("<H", stride))
            program.append(dtype)
            program.append(0)  # flags
            program.extend(struct.pack("<I", 0))  # padding

        # Instructions
        program.extend(self.code)

        return bytes(program)


def generate_input_matrix(shape: GemmShape) -> list[list[int]]:
    matrix: list[list[int]] = []
    for m in range(shape.m):
        row = []
        for k in range(shape.k):
            row.append(((m * 17 + k * 5 + 3) % 15) - 7)
        matrix.append(row)
    return matrix


def generate_weight_matrix(shape: GemmShape) -> list[list[int]]:
    matrix: list[list[int]] = []
    for k in range(shape.k):
        row = []
        for n in range(shape.n):
            row.append(((k * 11 + n * 3 + 1) % 9) - 4)
        matrix.append(row)
    return matrix


def generate_epilogue_params(
    shape: GemmShape,
) -> tuple[list[int], list[int], list[int]]:
    n_channels = shape.n
    bias = [((idx * 9 + 5) % 17) - 8 for idx in range(n_channels)]
    multiplier = [1 << 30 for _ in range(n_channels)]  # x0.5 after SRDHM
    shift = [max(4, shape.k.bit_length() - 2) + (idx & 1) for idx in range(n_channels)]
    return bias, multiplier, shift


def pack_input_tiles(matrix: list[list[int]], tile: int) -> bytes:
    values: list[int] = []
    for m_base in range(0, len(matrix), tile):
        for k in range(len(matrix[0])):
            for lane in range(tile):
                values.append(matrix[m_base + lane][k])
    return pack_i8(values)


def pack_weight_rows(matrix: list[list[int]]) -> bytes:
    values = [value for row in matrix for value in row]
    return pack_i8(values)


def pack_output_rows(matrix: list[list[int]]) -> bytes:
    values = [value for row in matrix for value in row]
    return pack_i8(values)


def _byte_as_i8(x: int) -> int:
    """Interpret one raw byte as signed int8 (memory layout)."""
    return struct.unpack("b", bytes([x & 0xFF]))[0]


def verify_gemm_output(
    expected: bytes,
    actual: bytes,
    *,
    shape: GemmShape,
    tolerance: int,
    mismatch_report_limit: int,
    dump_failed_hex_max: int,
) -> bool:
    """Compare device output to golden using signed i8 semantics.

    Older versions used ``abs(unsigned(e) - unsigned(a))``, which is wrong across
    the sign boundary (e.g. -1 vs 0 reads as 255). All errors are in signed i8 space.

    * ``tolerance=0`` (default): require exact match — use this to hunt integration/DMA 1-LSB drift.
    * ``tolerance=1``: allow at most one signed step per cell (legacy integration workaround).
    """
    if len(expected) != len(actual):
        print(
            f"[e2e] output size mismatch: expected {len(expected)}, got {len(actual)}"
        )
        return False

    if tolerance < 0 or tolerance > 127:
        raise ValueError("verify tolerance must be in 0..127")

    n = shape.n
    exact = 0
    off_by_one = 0
    max_abs = 0
    fail_indices: list[int] = []

    for i, (eb, ab) in enumerate(zip(expected, actual)):
        es = _byte_as_i8(eb)
        ac = _byte_as_i8(ab)
        delta = abs(es - ac)
        max_abs = max(max_abs, delta)
        if delta == 0:
            exact += 1
        elif delta == 1:
            off_by_one += 1
        if delta > tolerance:
            fail_indices.append(i)

    total = len(expected)
    print(
        f"[e2e] verify stats: {exact}/{total} exact, "
        f"{off_by_one} cell(s) with |Δ|=1, max |Δ|={max_abs}, strict tolerance={tolerance}",
        flush=True,
    )

    if not fail_indices:
        if max_abs == 0:
            print(
                "[e2e] output verification passed (exact signed i8 match)", flush=True
            )
        else:
            print(
                f"[e2e] output verification passed (tolerance={tolerance}, max |Δ|={max_abs})",
                flush=True,
            )
        return True

    reported = 0
    for i in fail_indices:
        if reported >= mismatch_report_limit:
            break
        m, col = divmod(i, n)
        es = _byte_as_i8(expected[i])
        ac = _byte_as_i8(actual[i])
        print(
            f"[e2e] mismatch cell ({m},{col}) linear={i}: expected {es}, got {ac}, |Δ|={abs(es - ac)}",
            flush=True,
        )
        reported += 1
    remaining = len(fail_indices) - reported
    if remaining > 0:
        print(f"[e2e] ... and {remaining} more failing cell(s)", flush=True)

    print(
        f"[e2e] output verification FAILED: {len(fail_indices)}/{total} cell(s) exceed tolerance {tolerance}",
        flush=True,
    )
    if tolerance == 0 and off_by_one > 0 and len(fail_indices) == off_by_one:
        print(
            "[e2e] hint: failures are all |Δ|=1 — typical of DMA/bridge timing vs golden; "
            "use --verify-tolerance 1 only while debugging hardware.",
            flush=True,
        )

    limit = min(len(expected), dump_failed_hex_max)
    if limit > 0:
        print(f"[e2e] expected[:{limit}].hex() = {expected[:limit].hex()}", flush=True)
        print(f"[e2e] actual[:{limit}].hex()   = {actual[:limit].hex()}", flush=True)
        if len(expected) > limit:
            print(
                f"[e2e] (truncated; full output is {len(expected)} bytes)", flush=True
            )

    return False


INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def ref_srdhm(a: int, b: int) -> int:
    """Saturating Rounding Doubling High Multiply — matches hardware."""
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    ab = a * b
    nudge = 1 << 30
    result = (ab + nudge) >> 31
    return max(INT32_MIN, min(INT32_MAX, result))


def ref_rdbpot(x: int, exponent: int) -> int:
    """Rounding Divide by Power of Two — matches hardware."""
    if exponent == 0:
        return x
    mask = (1 << exponent) - 1
    remainder = x & mask
    sign_bit = (x >> 31) & 1
    threshold = (mask >> 1) + sign_bit
    rounding = 1 if remainder > threshold else 0
    return (x >> exponent) + rounding


def compute_expected_output(
    input_matrix: list[list[int]],
    weight_matrix: list[list[int]],
    bias: list[int],
    multiplier: list[int],
    shift: list[int],
) -> bytes:
    """Compute expected GEMM output using the same epilogue math as hardware."""
    m_dim = len(input_matrix)
    k_dim = len(input_matrix[0])
    n_dim = len(weight_matrix[0])

    output: list[list[int]] = []
    for m in range(m_dim):
        row: list[int] = []
        for n in range(n_dim):
            acc = 0
            for k in range(k_dim):
                acc += input_matrix[m][k] * weight_matrix[k][n]

            val = acc + bias[n]
            val = ref_srdhm(val, multiplier[n])
            val = ref_rdbpot(val, shift[n])
            val += 0  # output_offset (signed, matches hardware)
            val = max(-128, min(127, val))  # clamp in signed i8
            row.append(val)
        output.append(row)

    return pack_output_rows(output)


def generate_gemm_program(
    shape: GemmShape,
    layout: MemoryLayout,
    cfu_word_bits: int = DEFAULT_CFU_WORD_BITS,
    cfu_store_depth_words: int = DEFAULT_CFU_STORE_DEPTH_WORDS,
) -> bytes:
    """Generate a tiled GEMM IR program for one output tile wide in N."""
    builder = ProgramBuilder()

    if cfu_word_bits % 32 != 0:
        raise ValueError(f"cfu_word_bits must be a multiple of 32, got {cfu_word_bits}")
    if cfu_word_bits % 8 != 0:
        raise ValueError(f"cfu_word_bits must be a multiple of 8, got {cfu_word_bits}")

    if cfu_store_depth_words <= 0:
        raise ValueError("cfu_store_depth_words must be positive")

    line_bytes = cfu_word_bits // 8
    if shape.tile != line_bytes:
        raise ValueError(
            f"tile size {shape.tile} must match scratchpad line width {line_bytes} bytes"
        )

    dma_beats_per_line = cfu_word_bits // 32
    k_tile = cfu_store_depth_words // dma_beats_per_line
    if k_tile <= 0:
        raise ValueError("scratchpad depth is too small for one K step")

    # Activations are packed in M-tiles so each K step is one scratchpad line.
    tid_input = builder.add_tensor(
        layout.input_addr,
        shape.m,
        shape.k * shape.tile,
        shape.k,
        DTYPE_I8,
    )
    tid_weights = builder.add_tensor(
        layout.weights_addr,
        shape.k,
        shape.n,
        shape.n,
        DTYPE_I8,
    )
    tid_output = builder.add_tensor(
        layout.output_addr,
        shape.m,
        shape.n,
        shape.n,
        DTYPE_I8,
    )
    tid_bias = builder.add_tensor(
        layout.bias_addr,
        1,
        shape.n,
        shape.n,
        DTYPE_I32,
    )
    tid_mult = builder.add_tensor(
        layout.mult_addr,
        1,
        shape.n,
        shape.n,
        DTYPE_I32,
    )
    tid_shift = builder.add_tensor(
        layout.shift_addr,
        1,
        shape.n,
        shape.n,
        DTYPE_I32,
    )

    # Each M tile reuses the same weight block and epilogue params while streaming
    # multiple K chunks through the double-buffered DMA/compute pipeline.
    for m_base in range(0, shape.m, shape.tile):
        builder.set_epilogue(
            bias_id=tid_bias,
            mult_id=tid_mult,
            shift_id=tid_shift,
            n_offset=0,
            n_count=shape.n,
            output_offset=0,
            act_min=-128,
            act_max=127,
        )

        for k_base in range(0, shape.k, k_tile):
            k_chunk = min(k_tile, shape.k - k_base)
            k_words = k_chunk * dma_beats_per_line
            builder.tile_load_act(
                tid_input,
                m_offset=m_base,
                k_offset=k_base * shape.tile,
                k_words=k_words,
            )
            builder.tile_load_wgt(
                tid_weights,
                n_offset=0,
                k_offset=k_base,
                k_words=k_words,
            )
            builder.tile_mma(
                first=(k_base == 0),
                last=(k_base + k_chunk == shape.k),
                k_count=k_chunk,
            )

        builder.tile_store(
            tid_output,
            m_offset=m_base,
            n_offset=0,
            m_count=shape.tile,
            n_count=shape.n,
        )

    builder.done()
    return builder.build()


def write_blob(
    driver: pathlib.Path,
    port: str,
    addr: int,
    blob: bytes,
    suffix: str,
    *,
    timeout_s: float,
) -> None:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(blob)
        handle.flush()
        temp_path = pathlib.Path(handle.name)
    try:
        run(
            [str(driver), port, "write-file", hex(addr), str(temp_path)],
            timeout_s=timeout_s,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a full GEMM end-to-end test against a board running the accel firmware.",
    )
    parser.add_argument(
        "port",
        nargs="?",
        default="/dev/ttyUSB1",
        help="Serial port, for example /dev/ttyUSB1.",
    )
    parser.add_argument(
        "--driver",
        default=str(DRIVER_BIN),
        help="Path to the built Zig driver executable.",
    )
    parser.add_argument(
        "--program",
        help="Optional: path to a pre-built IR program. If not provided, generates one.",
    )
    parser.add_argument(
        "--cfu-word-bits",
        type=int,
        default=DEFAULT_CFU_WORD_BITS,
        help="Scratchpad line width in bits (for example 64 on 8x8, 32 on 4x4).",
    )
    parser.add_argument(
        "--cfu-store-depth-words",
        type=int,
        default=DEFAULT_CFU_STORE_DEPTH_WORDS,
        help="Scratchpad depth in 32-bit words.",
    )
    parser.add_argument(
        "--m",
        type=int,
        help="Rows in the logical input/output matrix. Currently must match the array tile size.",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Shared inner dimension. Defaults to the array tile size.",
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Columns in the logical output matrix. Defaults to the array tile size.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip output verification (useful if input/weights not uploaded).",
    )
    parser.add_argument(
        "--verify-tolerance",
        type=int,
        default=0,
        help=(
            "Max allowed absolute error per output cell in signed i8 space (default: 0 = exact). "
            "Use 1 only to tolerate known integration/DMA 1-LSB drift while fixing hardware."
        ),
    )
    parser.add_argument(
        "--verify-report-limit",
        type=int,
        default=16,
        help="Max number of failing cells to print (coordinates + signed values).",
    )
    parser.add_argument(
        "--verify-failed-hex-bytes",
        type=int,
        default=128,
        help="On failure, dump this many leading bytes of expected/actual as hex (0 to disable).",
    )
    parser.add_argument(
        "--driver-timeout",
        type=float,
        default=DEFAULT_DRIVER_TIMEOUT_S,
        help=(
            "Seconds per driver subprocess (default: "
            f"{DEFAULT_DRIVER_TIMEOUT_S} or ACCEL_DRIVER_TIMEOUT_S). "
            "Prevents indefinite hang if the device does not respond."
        ),
    )
    args = parser.parse_args()

    driver = pathlib.Path(args.driver)

    if not driver.is_file():
        raise SystemExit(
            f"[e2e] missing driver binary: {driver}\n"
            "build it first with: zig build -Dbuild-dir=build/sipeed_tang_nano_20k"
        )

    tile = args.cfu_word_bits // 8
    shape = GemmShape(
        m=args.m or tile,
        k=args.k or tile,
        n=args.n or tile,
        tile=tile,
    )
    shape.validate()

    print(
        f"[e2e] generating self-contained GEMM data for M={shape.m}, K={shape.k}, N={shape.n}",
        flush=True,
    )
    input_matrix = generate_input_matrix(shape)
    weight_matrix = generate_weight_matrix(shape)
    bias, multiplier, shift = generate_epilogue_params(shape)

    input_data = pack_input_tiles(input_matrix, shape.tile)
    weight_data = pack_weight_rows(weight_matrix)
    bias_data = pack_i32(bias)
    mult_data = pack_i32(multiplier)
    shift_data = pack_i32(shift)

    layout = plan_memory(
        input_size=len(input_data),
        weight_size=len(weight_data),
        output_size=shape.m * shape.n,
        n_channels=shape.n,
    )

    # Compute expected output on CPU
    print("[e2e] computing expected output on CPU", flush=True)
    expected_output = compute_expected_output(
        input_matrix, weight_matrix, bias, multiplier, shift
    )

    # Upload data to device
    print("[e2e] uploading test data to device", flush=True)
    to = args.driver_timeout
    write_blob(
        driver, args.port, layout.input_addr, input_data, ".input.bin", timeout_s=to
    )
    write_blob(
        driver,
        args.port,
        layout.weights_addr,
        weight_data,
        ".weights.bin",
        timeout_s=to,
    )

    # Upload epilogue parameters
    print("[e2e] uploading epilogue parameters", flush=True)
    write_blob(
        driver, args.port, layout.bias_addr, bias_data, ".bias.bin", timeout_s=to
    )
    write_blob(
        driver, args.port, layout.mult_addr, mult_data, ".mult.bin", timeout_s=to
    )
    write_blob(
        driver, args.port, layout.shift_addr, shift_data, ".shift.bin", timeout_s=to
    )

    # Generate or load IR program
    if args.program:
        program = pathlib.Path(args.program).read_bytes()
        print(f"[e2e] loaded IR program from {args.program}", flush=True)
    else:
        program = generate_gemm_program(
            shape,
            layout,
            cfu_word_bits=args.cfu_word_bits,
            cfu_store_depth_words=args.cfu_store_depth_words,
        )
        print(f"[e2e] generated IR program ({len(program)} bytes)", flush=True)

    # Save program to temp file for driver
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_prog:
        tmp_prog.write(program)
        tmp_prog.flush()
        program_file = tmp_prog.name

    try:
        # Execute the IR program
        output = run([str(driver), args.port, "exec-bin", program_file], timeout_s=to)
        if "exec-bin ok" not in output:
            raise SystemExit("[e2e] firmware did not report a successful EXEC")

        print("[e2e] GEMM execution completed", flush=True)

        # Read and verify output
        if not args.no_verify:
            print("[e2e] reading and verifying output", flush=True)
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_out:
                output_file = tmp_out.name

            output_size = shape.m * shape.n
            run(
                [
                    str(driver),
                    args.port,
                    "read-file",
                    hex(layout.output_addr),
                    str(output_size),
                    output_file,
                ],
                timeout_s=to,
            )

            actual_output = pathlib.Path(output_file).read_bytes()
            pathlib.Path(output_file).unlink()

            if verify_gemm_output(
                expected_output,
                actual_output,
                shape=shape,
                tolerance=args.verify_tolerance,
                mismatch_report_limit=args.verify_report_limit,
                dump_failed_hex_max=args.verify_failed_hex_bytes,
            ):
                print("[e2e] full GEMM test PASSED", flush=True)
                return 0
            raise SystemExit("[e2e] GEMM output verification failed")
        else:
            print("[e2e] full GEMM test completed (verification skipped)", flush=True)
            return 0
    finally:
        pathlib.Path(program_file).unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
