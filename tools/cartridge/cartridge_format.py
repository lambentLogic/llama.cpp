"""Reader / writer for llama.cpp "state-seq" files (LLAMA_FILE_MAGIC_GGSQ).

This is the on-disk format produced by `llama_state_seq_save_file` and consumed by
`llama_state_seq_load_file` (see src/llama-context.cpp:2974 and
src/llama-kv-cache.cpp:1862). A KV "cartridge" is just one of these files whose K/V
rows were produced offline instead of by a live forward pass.

File layout (little-endian):

    u32  magic           = LLAMA_FILE_MAGIC_GGSQ ('ggsq', 0x67677371)
    u32  version         = LLAMA_STATE_SEQ_VERSION (2)
    u32  n_token_count
    i32  tokens[n_token_count]
    --- llama_kv_cache::state_write ---
    u32  n_stream
    per stream:
        u32 cell_count
        if cell_count > 0:
            meta: per cell { i32 pos; u32 n_seq_id; i32 seq_id[n_seq_id] }
                  (an extra llama_kv_cell_ext is written when n_pos_per_embd > 1,
                   e.g. mrope models — NOT supported here; we assert it is absent)
            data:
                u32 v_trans
                u32 n_layer
                per layer K: { i32 k_type; u64 k_size_row; raw[cell_count * k_size_row] }
                if v_trans == 0:
                    per layer V: { i32 v_type; u64 v_size_row; raw[cell_count * v_size_row] }
                else:
                    per layer V (transposed): { i32 v_type; u64 v_size_el;
                                                for each of n_embd_v_gqa rows: raw[cell_count * v_size_el] }

This module round-trips the format byte-for-byte and exposes the per-layer K/V as
numpy arrays for the converter and the test harness.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

LLAMA_FILE_MAGIC_GGSQ = 0x67677371  # 'ggsq'
LLAMA_STATE_SEQ_VERSION = 2

# ggml_type enum values we care about (ggml/include/ggml.h)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

_GGML_TYPE_SIZE = {GGML_TYPE_F32: 4, GGML_TYPE_F16: 2}
_GGML_TYPE_NP = {GGML_TYPE_F32: np.float32, GGML_TYPE_F16: np.float16}
_NP_GGML_TYPE = {"f32": GGML_TYPE_F32, "f16": GGML_TYPE_F16}


def ggml_row_size(ggml_type: int, n_elements: int) -> int:
    """Row size in bytes for an unquantized type (matches ggml_row_size)."""
    if ggml_type not in _GGML_TYPE_SIZE:
        raise ValueError(f"unsupported ggml_type {ggml_type} (only f32/f16 handled here)")
    return _GGML_TYPE_SIZE[ggml_type] * n_elements


@dataclass
class LayerKV:
    """One layer's K and V cache rows, as float arrays of shape (cell_count, n_embd_gqa)."""
    k_type: int
    v_type: int
    k: np.ndarray  # (cell_count, n_embd_k_gqa) float
    v: np.ndarray  # (cell_count, n_embd_v_gqa) float


@dataclass
class Cartridge:
    n_stream: int = 1
    stream_index: int = 0          # which stream carries the cells (rest are empty)
    cell_count: int = 0
    positions: list[int] = field(default_factory=list)  # len == cell_count
    seq_id: int = 0                # cosmetic; ignored on load (dest seq is chosen at load time)
    v_trans: int = 0
    tokens: list[int] = field(default_factory=list)
    layers: list[LayerKV] = field(default_factory=list)

    @property
    def n_layer(self) -> int:
        return len(self.layers)


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.off = 0

    def _take(self, n: int) -> bytes:
        b = self.data[self.off:self.off + n]
        if len(b) != n:
            raise EOFError("unexpected end of state-seq file")
        self.off += n
        return b

    def u32(self) -> int:
        return struct.unpack_from("<I", self._take(4))[0]

    def i32(self) -> int:
        return struct.unpack_from("<i", self._take(4))[0]

    def u64(self) -> int:
        return struct.unpack_from("<Q", self._take(8))[0]

    def raw(self, n: int) -> bytes:
        return self._take(n)


def read_cartridge(path: str) -> Cartridge:
    with open(path, "rb") as f:
        r = _Reader(f.read())

    magic = r.u32()
    if magic != LLAMA_FILE_MAGIC_GGSQ:
        raise ValueError(f"bad magic 0x{magic:08x} (expected ggsq 0x{LLAMA_FILE_MAGIC_GGSQ:08x})")
    version = r.u32()
    if version != LLAMA_STATE_SEQ_VERSION:
        raise ValueError(f"unsupported state-seq version {version} (expected {LLAMA_STATE_SEQ_VERSION})")

    n_token_count = r.u32()
    tokens = [r.i32() for _ in range(n_token_count)]

    cart = Cartridge(tokens=tokens)

    n_stream = r.u32()
    cart.n_stream = n_stream

    found = False
    for s in range(n_stream):
        cell_count = r.u32()
        if cell_count == 0:
            continue
        if found:
            raise ValueError("multiple populated streams not supported by this reader")
        found = True
        cart.stream_index = s
        cart.cell_count = cell_count

        # meta
        positions = []
        seq_id = 0
        for _ in range(cell_count):
            pos = r.i32()
            n_seq_id = r.u32()
            if n_seq_id != 1:
                raise ValueError(f"expected n_seq_id == 1 per cell, got {n_seq_id}")
            seq_id = r.i32()
            positions.append(pos)
        cart.positions = positions
        cart.seq_id = seq_id

        # data
        v_trans = r.u32()
        cart.v_trans = v_trans
        n_layer = r.u32()

        # K for every layer first, then V for every layer
        k_blocks = []
        for _ in range(n_layer):
            k_type = r.i32()
            k_size_row = r.u64()
            n_embd = k_size_row // _GGML_TYPE_SIZE[k_type]
            buf = r.raw(cell_count * k_size_row)
            arr = np.frombuffer(buf, dtype=_GGML_TYPE_NP[k_type]).reshape(cell_count, n_embd)
            k_blocks.append((k_type, arr.astype(np.float32)))

        v_blocks = []
        if v_trans == 0:
            for _ in range(n_layer):
                v_type = r.i32()
                v_size_row = r.u64()
                n_embd = v_size_row // _GGML_TYPE_SIZE[v_type]
                buf = r.raw(cell_count * v_size_row)
                arr = np.frombuffer(buf, dtype=_GGML_TYPE_NP[v_type]).reshape(cell_count, n_embd)
                v_blocks.append((v_type, arr.astype(np.float32)))
        else:
            raise NotImplementedError(
                "transposed V (v_trans=1) reading not implemented; run with flash attention (-fa) "
                "so the runtime stores V untransposed"
            )

        for (k_type, k), (v_type, v) in zip(k_blocks, v_blocks):
            cart.layers.append(LayerKV(k_type=k_type, v_type=v_type, k=k, v=v))

    if r.off != len(r.data):
        raise ValueError(f"trailing bytes after parse: parsed {r.off} of {len(r.data)}")

    return cart


def write_cartridge(path: str, cart: Cartridge) -> None:
    if cart.v_trans != 0:
        raise NotImplementedError("writing transposed V (v_trans=1) is not supported")
    if len(cart.positions) != cart.cell_count:
        raise ValueError("positions length must equal cell_count")

    out = bytearray()

    def u32(v):
        out.extend(struct.pack("<I", v))

    def i32(v):
        out.extend(struct.pack("<i", v))

    def u64(v):
        out.extend(struct.pack("<Q", v))

    u32(LLAMA_FILE_MAGIC_GGSQ)
    u32(LLAMA_STATE_SEQ_VERSION)
    u32(len(cart.tokens))
    for t in cart.tokens:
        i32(t)

    u32(cart.n_stream)
    for s in range(cart.n_stream):
        if s != cart.stream_index:
            u32(0)  # empty stream
            continue
        u32(cart.cell_count)

        # meta
        for pos in cart.positions:
            i32(pos)
            u32(1)            # n_seq_id
            i32(cart.seq_id)  # cosmetic; remapped to dest seq on load

        # data
        u32(cart.v_trans)
        u32(cart.n_layer)
        for layer in cart.layers:
            n_embd = layer.k.shape[1]
            i32(layer.k_type)
            u64(ggml_row_size(layer.k_type, n_embd))
            out.extend(layer.k.astype(_GGML_TYPE_NP[layer.k_type]).tobytes())
        for layer in cart.layers:
            n_embd = layer.v.shape[1]
            i32(layer.v_type)
            u64(ggml_row_size(layer.v_type, n_embd))
            out.extend(layer.v.astype(_GGML_TYPE_NP[layer.v_type]).tobytes())

    with open(path, "wb") as f:
        f.write(bytes(out))
