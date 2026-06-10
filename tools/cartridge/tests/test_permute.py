#!/usr/bin/env python3
"""Permute unit test (b): validate the rotate_half -> NORM-rope K permutation and the
head-major cache layout by comparing llama.cpp's cached K/V against an HF forward pass.

Pipeline:
  1. Read a ground-truth cartridge produced by:
        llama-cartridge-cli -m <llama/mistral.gguf> --save-cartridge gt.cartridge -p "<text>"
     The file carries both the exact token ids llama used and llama's post-rope K/V.
  2. Run those exact token ids through the matching HF model (use_cache=True) to get HF's
     post-rope past_key_values (positions 0..N-1).
  3. Apply the converter's permute to HF K, lay out head-major, and compare to llama K.
     V is compared without the permute.

We report, per layer, the cosine similarity and relative L2 error of the CORRECT mapping,
and (as a control) the cosine of the WRONG mapping (no permute on K). The correct mapping
should be ~1.0 and the control should be clearly lower — that is the actual evidence that
the permute is required and right. Small residual error is expected from weight
quantization in the GGUF.

Usage:
    python3 tools/cartridge/tests/test_permute.py <gt.cartridge> <hf_model_dir>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cartridge_format as cf  # noqa: E402
from convert_cartridge import permute_k_head_dim  # noqa: E402


def _hf_layer_kv(pkv, i):
    """Return (K, V) for layer i as numpy (n_kv, seq, head_dim), across transformers versions."""
    import torch

    def np3(t):
        a = t.detach().to(torch.float32).cpu().numpy()
        return a[0] if a.ndim == 4 else a  # drop batch

    if hasattr(pkv, "layers"):
        layer = pkv.layers[i]
        return np3(layer.keys), np3(layer.values)
    if hasattr(pkv, "key_cache"):
        return np3(pkv.key_cache[i]), np3(pkv.value_cache[i])
    return np3(pkv[i][0]), np3(pkv[i][1])  # tuple form


def cos(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def relerr(a, b):
    return float(np.linalg.norm((a - b).ravel()) / (np.linalg.norm(b.ravel()) + 1e-12))


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    cart_path, hf_dir = sys.argv[1], sys.argv[2]

    import torch
    from transformers import AutoModelForCausalLM

    cart = cf.read_cartridge(cart_path)
    ids = cart.tokens
    n = cart.cell_count
    print(f"cartridge: {cart.n_layer} layers, {n} tokens, ids={ids}")

    print("loading HF model (float32, cpu) ...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=True)
    pkv = out.past_key_values

    head_dim = _hf_layer_kv(pkv, 0)[0].shape[-1]
    print(f"HF head_dim={head_dim}")

    cos_k_correct, cos_k_wrong, rel_k, cos_v, rel_v = [], [], [], [], []
    for il in range(cart.n_layer):
        hk, hv = _hf_layer_kv(pkv, il)          # (n_kv, seq, head_dim)
        n_kv = hk.shape[0]

        # llama rows -> (seq, n_kv, head_dim)
        lk = cart.layers[il].k.reshape(n, n_kv, head_dim)
        lv = cart.layers[il].v.reshape(n, n_kv, head_dim)

        # HF -> (seq, n_kv, head_dim)
        hk_t = np.transpose(hk, (1, 0, 2))
        hv_t = np.transpose(hv, (1, 0, 2))

        hk_perm = permute_k_head_dim(hk_t, head_dim)  # correct mapping

        cos_k_correct.append(cos(hk_perm, lk))
        cos_k_wrong.append(cos(hk_t, lk))             # control: no permute
        rel_k.append(relerr(hk_perm, lk))
        cos_v.append(cos(hv_t, lv))
        rel_v.append(relerr(hv_t, lv))

    ckc = float(np.mean(cos_k_correct))
    ckw = float(np.mean(cos_k_wrong))
    rk = float(np.mean(rel_k))
    cv = float(np.mean(cos_v))
    rv = float(np.mean(rel_v))

    print("\n--- per-layer means ---")
    print(f"K cos (permuted, correct) : {ckc:.5f}   relL2 {rk:.4f}")
    print(f"K cos (no-permute, control): {ckw:.5f}")
    print(f"V cos                      : {cv:.5f}   relL2 {rv:.4f}")
    print(f"\nlayer0 K cos correct={cos_k_correct[0]:.5f} wrong={cos_k_wrong[0]:.5f}")
    print(f"layer{cart.n_layer-1} K cos correct={cos_k_correct[-1]:.5f} wrong={cos_k_wrong[-1]:.5f}")

    # Pass criteria: permuted K and V both highly correlated, and the permute clearly beats
    # the control. Tolerances are loose because the GGUF is quantized.
    ok = ckc > 0.99 and cv > 0.99 and (ckc - ckw) > 0.2
    print("\n" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
