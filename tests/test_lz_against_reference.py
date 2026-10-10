"""
tests/test_lz_against_reference.py

Checks src/decoding.py:LZPenaltyDecoder._penalty_vector against the penalty
function of the authors' reference implementation, pinned to
tginart/sglang commit ab08e6976d4980e350df3891e40967a9718fae8b
(python/sglang/srt/sampling/penaltylib/lz_penalty.py, 2025-07-01).

The reference file imports the sglang runtime at module level, so the three
pure-tensor functions it needs (compute_lz_encoding_cost,
compute_reversed_consecutive_counts, compute_lz_penalty_stateless) are
extracted from its source and executed in isolation.  One compatibility
patch is applied to the extracted source: compute_lz_encoding_cost receives
an integer tensor and calls torch.full_like(x, float('inf')), which raises
on current PyTorch; the patch casts x to float first (the values are
unchanged).  The reference applies   logits.sub_(lz_penalty * penalty)   with
penalty = log2(V) - cost; this project adds alpha * (cost - log2(V)) to the
logits, so the comparison is   ours == -reference.

What is and is not tested: the penalty VECTOR for a given token history is
compared exactly.  The history handed to the penalty differs by design: the
reference keeps only generated tokens in a window of lz_lookback_size, while
LZPenaltyDecoder keeps prompt and generated tokens and uses the last
buffer_size + window_size of them.  This test therefore feeds both the same
context explicitly.

The reference carries an Apache-2.0 license; its code is fetched at run time
(cached under tests/.official_cache/, git-ignored) and not vendored.

Run:  python3 tests/test_lz_against_reference.py
"""
import ast, math, os, random, sys, urllib.request

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.decoding import LZPenaltyDecoder  # noqa: E402

COMMIT = "ab08e6976d4980e350df3891e40967a9718fae8b"
URL = ("https://raw.githubusercontent.com/tginart/sglang/"
       f"{COMMIT}/python/sglang/srt/sampling/penaltylib/lz_penalty.py")
CACHE = os.path.join(ROOT, "tests", ".official_cache", f"lz_penalty_{COMMIT[:10]}.py")
WANTED = ("compute_lz_encoding_cost", "compute_reversed_consecutive_counts",
          "compute_lz_penalty_stateless")


def load_reference():
    if not os.path.exists(CACHE):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        try:
            with urllib.request.urlopen(URL, timeout=20) as r:
                open(CACHE, "wb").write(r.read())
        except Exception as e:
            print(f"SKIP: could not fetch the LZ reference ({e})")
            return None
    src = open(CACHE).read()
    tree = ast.parse(src)
    chunks = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in WANTED:
            chunk = ast.get_source_segment(src, node)
            if node.name == "compute_lz_encoding_cost":
                chunk = chunk.replace(
                    "    if x.numel() == 0:",
                    "    x = x.float()  # compatibility patch (see test docstring)\n"
                    "    if x.numel() == 0:", 1)
            chunks.append(chunk)
    ns = {"torch": torch, "math": math}
    exec("\n\n".join(chunks), ns)
    return ns["compute_lz_penalty_stateless"]


def ours(context, vocab, n):
    dec = LZPenaltyDecoder(buffer_size=n, window_size=max(len(context), 1))
    dec.reset()
    dec.prime(list(context))
    return dec._penalty_vector(vocab)


def compare(ref_fn, context, vocab, n):
    ref = ref_fn(torch.tensor(context, dtype=torch.long),
                 torch.arange(vocab, dtype=torch.long), n)
    got = ours(context, vocab, n)
    err = float((got + ref).abs().max())    # ours == -reference
    assert err == 0.0, (context, n, err)
    return got


def test_constructed(ref_fn):
    compare(ref_fn, [1, 2, 3, 4, 2, 3, 4, 5], 10, 3)           # reference's own demo
    compare(ref_fn, [1, 2, 3, 4, 1, 2, 3, 5], 10, 2)
    compare(ref_fn, [5, 6, 7, 8, 5, 6, 7, 9], 12, 3)
    compare(ref_fn, [9, 8, 7, 6, 9, 8, 7, 5], 12, 4)
    # Context no longer than n: the reference's early return passes the vocab
    # TENSOR to torch.zeros as a size (a bug on that path, it raises), so only
    # the intended behaviour (all-zero penalty) is checked, on our side.
    assert float(ours([1, 2], 10, 3).abs().sum()) == 0.0
    compare(ref_fn, [3] * 20, 6, 4)                              # period-1 run


def test_random(ref_fn, trials=300):
    rng = random.Random(0)
    for _ in range(trials):
        vocab = rng.choice([6, 12, 30])
        n = rng.choice([2, 3, 4, 8])
        length = rng.randint(n + 1, 120)
        context = [rng.randrange(vocab) for _ in range(length)]
        compare(ref_fn, context, vocab, n)


if __name__ == "__main__":
    ref_fn = load_reference()
    if ref_fn is None:
        sys.exit(0)
    test_constructed(ref_fn)
    test_random(ref_fn)
    print(f"OK: LZPenaltyDecoder._penalty_vector equals the reference "
          f"compute_lz_penalty_stateless (commit {COMMIT[:10]}) exactly on constructed "
          f"and 300 random contexts (max difference 0.0).")
