#!/usr/bin/env python3
"""Convert a HazyResearch "Cartridges" trained KV cache into a llama.cpp state-seq file.

A cartridge (cartridges/cache.py `TrainableCache`) is a `torch.save` dict of per-layer
K and V parameters, each shaped (1, n_kv_heads, n_tokens, head_dim), living in HF
post-RoPE space (rotate_half layout). llama.cpp also caches post-RoPE K, but for
NORM-rope archs (Llama / Mistral) the head_dim is laid out interleaved rather than
split-in-halves. We therefore apply the SAME permutation that GGUF conversion applies to
wq/wk (conversion/llama.py:114, `LlamaModel.permute`) to the head_dim axis of K. V is not
rotated, so V is passed through unchanged. The result is written as a state-seq file that
`llama_state_seq_load_file` injects directly into a sequence's KV cache at positions
0..N-1.

Assumptions (Phase 1, Mistral/Llama):
  * runtime uses flash attention (-fa)  -> V stored untransposed (v_trans=0)
  * runtime uses --kv-unified           -> single KV stream (n_stream=1)
  * cache dtype matches --cache-type (default f16)
  * the cartridge was trained against weights matching the target GGUF, with the same
    rope config (freq_base / scaling / head_dim)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import cartridge_format as cf


def _to_numpy(t):
    import torch  # local import so the module loads without torch for the format tests
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    return t.detach().to(torch.float32).cpu().numpy()


def _stack_layers(obj):
    """Normalize a per-layer container (ParameterList / list / stacked tensor) to a list
    of numpy arrays, each squeezed to (n_kv_heads, n_tokens, head_dim)."""
    import torch

    if isinstance(obj, torch.Tensor) and obj.dim() == 5:
        # (n_layer, 1, n_kv, N, hd)
        layers = [obj[i] for i in range(obj.shape[0])]
    else:
        layers = list(obj)  # ParameterList / list of tensors

    out = []
    for t in layers:
        a = _to_numpy(t)
        a = np.squeeze(a, axis=0) if a.ndim == 4 and a.shape[0] == 1 else a
        if a.ndim != 3:
            raise ValueError(f"expected per-layer KV of shape (n_kv, N, head_dim), got {a.shape}")
        out.append(a)
    return out


def _concat_frozen_trainable(state, key_base):
    """Return per-layer arrays of frozen ++ trainable tokens for 'keys' or 'values'."""
    frozen = state.get(f"frozen_{key_base}")
    trainable = state.get(f"trainable_{key_base}")
    if trainable is None:
        # some checkpoints store just "keys"/"values"
        trainable = state.get(key_base)
    if trainable is None:
        raise KeyError(f"cartridge has neither trainable_{key_base} nor {key_base}")

    tr = _stack_layers(trainable)
    if frozen is None:
        return tr
    fr = _stack_layers(frozen)
    if len(fr) != len(tr):
        raise ValueError("frozen/trainable layer count mismatch")
    return [np.concatenate([f, t], axis=1) for f, t in zip(fr, tr)]  # concat along token axis


def permute_k_head_dim(k: np.ndarray, head_dim: int) -> np.ndarray:
    """HF rotate_half layout -> ggml NORM (interleaved) layout, on the last axis.

    Mirrors conversion/llama.py LlamaModel.permute applied per head_dim block:
        reshape(..., 2, head_dim//2).swapaxes(-2, -1).reshape(..., head_dim)
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for NORM rope permute")
    shp = k.shape
    k = k.reshape(*shp[:-1], 2, head_dim // 2)
    k = np.swapaxes(k, -1, -2)
    return np.ascontiguousarray(k).reshape(*shp)


def build_cartridge(
    keys_per_layer: list[np.ndarray],
    values_per_layer: list[np.ndarray],
    cache_type: str,
    rope_type: str,
    pad_token: int,
    tokens: list[int] | None,
) -> cf.Cartridge:
    n_layer = len(keys_per_layer)
    if len(values_per_layer) != n_layer:
        raise ValueError("keys/values layer count mismatch")

    n_kv, n_tok, head_dim = keys_per_layer[0].shape
    ggml_type = cf._NP_GGML_TYPE[cache_type]

    cart = cf.Cartridge(
        n_stream=1,
        stream_index=0,
        cell_count=n_tok,
        positions=list(range(n_tok)),
        seq_id=0,
        v_trans=0,
    )

    if tokens is not None:
        if len(tokens) != n_tok:
            raise ValueError(f"--tokens length {len(tokens)} != cartridge token count {n_tok}")
        cart.tokens = list(tokens)
    else:
        cart.tokens = [pad_token] * n_tok

    for il in range(n_layer):
        k = keys_per_layer[il]    # (n_kv, N, head_dim)
        v = values_per_layer[il]
        if k.shape != (n_kv, n_tok, head_dim):
            raise ValueError(f"layer {il} K shape {k.shape} != {(n_kv, n_tok, head_dim)}")
        if v.shape[0] != n_kv or v.shape[1] != n_tok:
            raise ValueError(f"layer {il} V shape {v.shape} inconsistent")

        if rope_type == "norm":
            k = permute_k_head_dim(k, head_dim)
        elif rope_type != "neox":
            raise ValueError(f"unknown rope_type {rope_type}")

        # head-major flatten to one cache row per token: (N, n_kv*head_dim)
        k_rows = np.ascontiguousarray(k.transpose(1, 0, 2)).reshape(n_tok, -1)
        v_rows = np.ascontiguousarray(v.transpose(1, 0, 2)).reshape(n_tok, v.shape[0] * v.shape[2])

        cart.layers.append(cf.LayerKV(
            k_type=ggml_type, v_type=ggml_type,
            k=k_rows.astype(np.float32), v=v_rows.astype(np.float32),
        ))

    return cart


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cartridge", help="path to the trained cartridge (torch.save dict)")
    ap.add_argument("-o", "--out", required=True, help="output .cartridge (state-seq) file")
    ap.add_argument("--cache-type", choices=["f16", "f32"], default="f16",
                    help="KV cache dtype of the target runtime (must match --cache-type-k/v)")
    ap.add_argument("--rope-type", choices=["norm", "neox"], default="norm",
                    help="norm (Llama/Mistral, applies head_dim permute) or neox (Qwen/Gemma, no permute)")
    ap.add_argument("--pad-token", type=int, default=0,
                    help="token id used to fill the file's token list (Phase-2 bookkeeping only)")
    args = ap.parse_args(argv)

    import torch
    state = torch.load(args.cartridge, map_location="cpu", weights_only=False)
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    if not isinstance(state, dict):
        raise SystemExit("unexpected cartridge object; expected a dict of K/V parameters")

    keys = _concat_frozen_trainable(state, "keys")
    values = _concat_frozen_trainable(state, "values")

    n_kv, n_tok, head_dim = keys[0].shape
    print(f"cartridge: n_layer={len(keys)} n_kv_heads={n_kv} n_tokens={n_tok} head_dim={head_dim} "
          f"cache_type={args.cache_type} rope={args.rope_type}", file=sys.stderr)

    cart = build_cartridge(keys, values, args.cache_type, args.rope_type, args.pad_token, None)
    cf.write_cartridge(args.out, cart)
    print(f"wrote {args.out}  ({len(cart.layers)} layers, {cart.cell_count} cells)", file=sys.stderr)


if __name__ == "__main__":
    main()
