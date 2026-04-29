"""Memory layout helpers for packing tensors into hardware scratchpad format."""

import numpy as np


def pack_input_tiles(matrix: np.ndarray, tile: int) -> bytes:
    """Pack [M, K] activation matrix into [K, tile] scratchpad layout.

    The hardware expects [K, tile] column-major layout per M-tile.
    For each M-tile, we create K rows of tile values each.
    """
    m, _k = matrix.shape
    chunks: list[np.ndarray] = []
    for m_base in range(0, m, tile):
        tile_slice = matrix[m_base : m_base + tile, :].T  # [K, tile]
        if tile_slice.shape[1] < tile:
            pad = tile - tile_slice.shape[1]
            tile_slice = np.pad(tile_slice, ((0, 0), (0, pad)), mode="constant")
        chunks.append(np.ascontiguousarray(tile_slice))
    return np.concatenate(chunks).astype(np.int8).tobytes()


def pack_weight_rows(matrix: np.ndarray, tile: int = 8) -> bytes:
    """Pack [K, N] weight matrix into [K, tile] words expected by HW scratchpad."""
    k, n = matrix.shape
    words: list[bytes] = []
    for t in range(0, n, tile):
        for kk in range(k):
            vals = matrix[kk, t:t + tile]
            if vals.shape[0] < tile:
                vals = np.pad(vals, (0, tile - vals.shape[0]), mode="constant")
            words.append(np.ascontiguousarray(vals, dtype=np.int8).tobytes())
    return b"".join(words)


def align_up(value: int, alignment: int = 32) -> int:
    """Round value up to the next multiple of alignment."""
    return (value + alignment - 1) & -(alignment)
