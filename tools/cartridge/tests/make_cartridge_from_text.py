#!/usr/bin/env python3
"""Build a `torch.save` cartridge from a document's HF KV cache.

This reproduces the reference's `KVCacheFromPretrained` initialization: run the document
through the HF model and store the resulting per-layer post-RoPE K/V as the cartridge
parameters. It is a real, untrained cartridge -- enough to exercise the full converter and
to demonstrate that injecting it makes the document's content available to the model
without putting the document in the prompt.

Output is a torch.save dict in the reference shape:
    { "trainable_keys":  [ (1, n_kv, N, head_dim) per layer ],
      "trainable_values":[ (1, n_kv, N, head_dim) per layer ] }

Usage:
    python3 make_cartridge_from_text.py <hf_dir> <out.pt> --text "..."   (or --file doc.txt)
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("out")
    ap.add_argument("--text", default=None)
    ap.add_argument("--file", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.file:
        with open(args.file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("provide --text or --file", file=sys.stderr)
        return 2

    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    ids = tok(text, return_tensors="pt").input_ids
    print(f"document tokens: {ids.shape[1]}", file=sys.stderr)

    model = AutoModelForCausalLM.from_pretrained(args.hf_dir, dtype="float32", device_map="cpu")
    model.eval()
    with torch.no_grad():
        pkv = model(input_ids=ids, use_cache=True).past_key_values

    n_layer = model.config.num_hidden_layers
    keys, values = [], []
    for il in range(n_layer):
        if hasattr(pkv, "layers"):
            k, v = pkv.layers[il].keys, pkv.layers[il].values
        elif hasattr(pkv, "key_cache"):
            k, v = pkv.key_cache[il], pkv.value_cache[il]
        else:
            k, v = pkv[il][0], pkv[il][1]
        keys.append(k.to(torch.float16).contiguous())     # (1, n_kv, N, head_dim)
        values.append(v.to(torch.float16).contiguous())

    torch.save({"trainable_keys": keys, "trainable_values": values}, args.out)
    print(f"wrote {args.out}: {n_layer} layers, K[0]={tuple(keys[0].shape)}", file=sys.stderr)


if __name__ == "__main__":
    main()
