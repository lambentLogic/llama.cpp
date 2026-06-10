# KV Cartridges (inference loader)

Load a pre-trained **KV cartridge** — a trained KV cache prefix from the HazyResearch
[*Cartridges*](https://arxiv.org/abs/2506.06266) paper — into llama.cpp and generate with
it as cheap long-context, instead of putting the corpus in the prompt.

A cartridge is literally a set of per-layer K/V vectors for *N* "cartridge tokens"
(`torch.save` dict of `trainable_keys`/`trainable_values`, each `(1, n_kv_heads, N, head_dim)`,
in HF post-RoPE space). This tool converts one into a llama.cpp **state-seq file** and
injects it via the stock `llama_state_seq_load_file` API.

**Phase 1 scope:** inference only, **Mistral / Llama** (NORM rope). Training stays in the
HazyResearch PyTorch repo.

## Components

| File | What it does |
|------|--------------|
| `cartridge_format.py` | read/write the llama.cpp state-seq (`ggsq`) file format |
| `convert_cartridge.py` | `torch.save` cartridge → `.cartridge` state-seq file |
| `cartridge-cli.cpp` | `llama-cartridge-cli`: inject a cartridge and generate (or `--save-cartridge`) |
| `tests/` | format round-trip, permute validation, end-to-end demo |

## Usage

```bash
# 1. convert a trained cartridge (Mistral/Llama)
python3 tools/cartridge/convert_cartridge.py cartridge.pt -o cartridge.cartridge \
    --cache-type f16 --rope-type norm

# 2. generate with it (the corpus is NOT in the prompt)
./build/bin/llama-cartridge-cli -m model.gguf --cartridge cartridge.cartridge \
    --no-bos -p "Question: ... Answer:" -n 128
```

### Required runtime layout

The converter targets a specific cache layout, and `llama-cartridge-cli` forces it:

- **Flash attention enabled** → V stored untransposed (`v_trans = 0`).
- **Unified KV cache** (`--kv-unified`) → single KV stream (`n_stream = 1`).
- **Cache dtype** matches `--cache-type` (default `f16`; must equal `--cache-type-k/-v`).

The cartridge must be trained against weights matching the GGUF, with the **same rope
config** (`freq_base`, scaling, `head_dim`). Standard Mistral/Llama only — no SWA, uniform
`n_head_kv` (constant `n_embd_k_gqa`). The converter rejects other cases.

## The one transform

llama.cpp caches post-RoPE K, same space the cartridge lives in, so K/V are injected as-is —
**except** NORM-rope archs lay out `head_dim` interleaved, while HF uses `rotate_half`
(split halves). `convert_cartridge.py` applies the same permutation that GGUF conversion
applies to `wq/wk` (`conversion/llama.py` `LlamaModel.permute`) to the K `head_dim` axis.
V is not rotated, so V is untouched. For NEOX archs (Qwen/Gemma) pass `--rope-type neox`
(no permute) — untested in Phase 1.

## Tests (verified on Mistral Nemo 12B / Arsenic v4.4)

```bash
# (a) format round-trip — byte-exact
./build/bin/llama-cartridge-cli -m mistral.gguf -c 1024 \
    -p "The quick brown fox." --save-cartridge /tmp/gt.cartridge
python3 tools/cartridge/tests/test_format_roundtrip.py /tmp/gt.cartridge
#   -> PASS: byte-identical round-trip

# (b) permute vs a real HF forward
./build/bin/llama-cartridge-cli -m merged.gguf -c 512 \
    -p "..." --save-cartridge /tmp/c.cartridge
python3 tools/cartridge/tests/test_permute.py /tmp/c.cartridge <hf_dir>
#   -> K cos (permuted) 0.9999, control 0.019, V cos 0.9996  -> PASS

# (c) end-to-end: a cartridge built from a document supplies facts not in the prompt
python3 tools/cartridge/tests/make_cartridge_from_text.py <hf_dir> /tmp/doc.pt --text "<document>"
python3 tools/cartridge/convert_cartridge.py /tmp/doc.pt -o /tmp/doc.cartridge
./build/bin/llama-cartridge-cli -m merged.gguf --cartridge /tmp/doc.cartridge \
    --no-bos -p "Question: ... Answer:" -n 40
#   -> answers from the document; no-cartridge baseline hallucinates
```

## Not yet (later phases)

- **Server integration** — load a cartridge into a slot before the prompt; prepend the
  recorded sentinel tokens to `slot.prompt.tokens` + `input_tokens` so the common-prefix /
  eviction logic treats the cartridge as immutable context (see
  `tools/server/server-context.cpp` `SLOT_RESTORE`). Cartridge composition lands here too.
- **Native API + transposed-V / multi-stream** converter support (drop the `-fa` /
  `--kv-unified` requirement).
- **NEOX archs** (Qwen/Gemma) validation.
