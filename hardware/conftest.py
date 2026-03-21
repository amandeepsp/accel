"""Shared test helpers for CFU hardware simulation tests.

Run all tests:  uv run pytest hardware/ -v
"""

import struct

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def pack_bytes(a, b, c, d):
    """Pack 4 signed int8 values into a 32-bit unsigned int (little-endian)."""
    return struct.unpack("<I", struct.pack("<4b", a, b, c, d))[0]


def ref_srdhm(a: int, b: int) -> int:
    """Python reference implementation of SaturatingRoundingDoubleHighMul."""
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    ab = a * b
    nudge = 1 << 30
    result = (ab + nudge) >> 31
    return max(INT32_MIN, min(INT32_MAX, result))


def ref_rdbpot(x: int, exponent: int) -> int:
    """Python reference implementation of RoundingDivideByPowerOfTwo."""
    if exponent == 0:
        return x
    mask = (1 << exponent) - 1
    remainder = x & mask
    sign_bit = (x >> 31) & 1
    threshold = (mask >> 1) + sign_bit
    rounding = 1 if remainder > threshold else 0
    return (x >> exponent) + rounding


def to_signed32(val):
    """Convert unsigned 32-bit to signed 32-bit Python int."""
    if val >= (1 << 31):
        val -= 1 << 32
    return val
