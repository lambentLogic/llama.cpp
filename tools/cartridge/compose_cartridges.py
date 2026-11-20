#!/usr/bin/env python3
"""Compose (concatenate) multiple cartridges into one, with RoPE position-shifting.

Each cartridge was trained/extracted as the prefix at positions 0..N-1, so its keys are
post-RoPE for those positions. To stack cartridge #k after the earlier ones, its keys must
be re-roped by the cumulative offset (delta = sum of earlier lengths). RoPE is a rotation
and R(p+delta) = R(delta)·R(p), so we apply R(delta) to the already-baked keys — exact, no
retraining. Values are not roped, so they pass through unchanged.

This operates in llama.cpp NORM-rope layout (the .cartridge K layout): the head_dim is in
adjacent-pair order, pair k = dims (2k, 2k+1), frequency = freq_base^(-2k/head_dim). Only
valid for NORM-rope archs (Llama/Mistral) with full rotary (n_rot == head_dim).

Usage:
    compose_cartridges.py A.cartridge B.cartridge [C.cartridge ...] -o combined.cartridge \\
        --head-dim 128 --rope-freq-base 1000000
"""
from __future__ import annotations

import argparse

import numpy as np

import cartridge_format as cf


def rope_shift_norm(k_rows: np.ndarray, n_kv: int, head_dim: int, delta: int, freq_base: float) -> np.ndarray:
    """Apply a NORM-rope rotation of `delta` positions to K rows (cell_count, n_kv*head_dim)."""
    if delta == 0:
        return k_rows
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")
    n = k_rows.shape[0]
    k = k_rows.astype(np.float64).reshape(n, n_kv, head_dim)
    half = head_dim // 2
    kpair = np.arange(half)
    freq = freq_base ** (-2.0 * kpair / head_dim)      # (half,)
    angle = delta * freq                                # (half,)
    cos, sin = np.cos(angle), np.sin(angle)
    even = k[..., 0::2]                                  # dims 0,2,4,...  -> pair lo
    odd = k[..., 1::2]                                   # dims 1,3,5,...  -> pair hi
    new_even = even * cos - odd * sin
    new_odd = even * sin + odd * cos
    out = np.empty_like(k)
    out[..., 0::2] = new_even
    out[..., 1::2] = new_odd
    return out.reshape(n, n_kv * head_dim).astype(k_rows.dtype)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="cartridge files to concatenate, in order")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--head-dim", type=int, required=True)
    ap.add_argument("--rope-freq-base", type=float, default=1000000.0)
    args = ap.parse_args(argv)

    carts = [cf.read_cartridge(p) for p in args.inputs]
    n_layer = carts[0].n_layer
    for c in carts:
        if c.n_layer != n_layer:
            raise SystemExit("layer count mismatch across cartridges")
        if c.v_trans != 0:
            raise SystemExit("only v_trans=0 cartridges supported")

    n_embd = carts[0].layers[0].k.shape[1]
    n_kv = n_embd // args.head_dim

    combined = cf.Cartridge(n_stream=1, stream_index=0, v_trans=0)
    offset = 0
    all_tokens = []
    for c in carts:
        all_tokens += c.tokens
    combined.tokens = all_tokens

    # build per-layer concatenated K/V with position-shift on K
    per_layer_k = [[] for _ in range(n_layer)]
    per_layer_v = [[] for _ in range(n_layer)]
    total_cells = 0
    for c in carts:
        n = c.cell_count
        for il in range(n_layer):
            k = rope_shift_norm(c.layers[il].k, n_kv, args.head_dim, offset, args.rope_freq_base)
            per_layer_k[il].append(k)
            per_layer_v[il].append(c.layers[il].v)
        offset += n
        total_cells += n

    combined.cell_count = total_cells
    combined.positions = list(range(total_cells))
    ktype = carts[0].layers[0].k_type
    vtype = carts[0].layers[0].v_type
    for il in range(n_layer):
        combined.layers.append(cf.LayerKV(
            k_type=ktype, v_type=vtype,
            k=np.concatenate(per_layer_k[il], axis=0),
            v=np.concatenate(per_layer_v[il], axis=0),
        ))

    cf.write_cartridge(args.out, combined)
    lens = [c.cell_count for c in carts]
    print(f"composed {len(carts)} cartridges {lens} -> {total_cells} cells, {n_layer} layers -> {args.out}")


if __name__ == "__main__":
    main()
