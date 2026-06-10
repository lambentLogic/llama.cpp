#!/usr/bin/env python3
"""Format self-test (a): prove the Python reader/writer round-trips a real llama.cpp
state-seq file byte-for-byte.

Generate a ground-truth file first with a NON-SWA Mistral/Llama model:

    ./build/bin/llama-cartridge-cli -m <mistral_or_llama.gguf> -ngl 99 -c 1024 \\
        -p "The quick brown fox jumps over the lazy dog." \\
        --save-cartridge /tmp/gt.cartridge

then:

    python3 tools/cartridge/tests/test_format_roundtrip.py /tmp/gt.cartridge

Exits non-zero on any mismatch.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cartridge_format as cf  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = sys.argv[1]

    cart = cf.read_cartridge(path)
    print(f"parsed: n_stream={cart.n_stream} stream={cart.stream_index} cells={cart.cell_count} "
          f"n_layer={cart.n_layer} v_trans={cart.v_trans} tokens={len(cart.tokens)} "
          f"n_embd_k_gqa={cart.layers[0].k.shape[1]} k_type={cart.layers[0].k_type}")

    tmp = path + ".roundtrip"
    cf.write_cartridge(tmp, cart)

    with open(path, "rb") as f:
        a = f.read()
    with open(tmp, "rb") as f:
        b = f.read()
    os.remove(tmp)

    if a != b:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"FAIL: first byte diff at offset {i}: {x} != {y}")
                return 1
        print(f"FAIL: length differs {len(a)} != {len(b)}")
        return 1

    print(f"PASS: byte-identical round-trip ({len(a)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
