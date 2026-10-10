"""
tests/test_fsd_against_official.py

Checks src/decoding.py:FSDDecoder._anti_lm_scores against the anti-LM penalty
of the official FSD implementation (github.com/LHRYANG/FSD,
ngram_model/fsd.py, NGram.penalize) on constructed and random prefixes.

The official repository ships no license, so its code is not vendored here:
the file is downloaded at run time and cached under tests/.official_cache/
(git-ignored). If it cannot be fetched, the tests are skipped, not passed.

Run:  python3 tests/test_fsd_against_official.py
"""
import importlib.util, os, random, sys, urllib.request

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.decoding import FSDDecoder  # noqa: E402

# Pinned: LHRYANG/FSD master at 46df7d5bc6228bd5045b68c09f85d75bcceb1fdf
# (ngram_model/fsd.py last changed in 0f469ac54f26b02d20f831a827c09336ede5f06a,
# 2024-02-23).  Note: that commit's utils.py calls NGram objects directly
# (ng_list[i](...)), which the pinned fsd.py does not define; the penalty is
# computed by NGram.penalize, which is what this test compares against.
OFFICIAL_COMMIT = "46df7d5bc6228bd5045b68c09f85d75bcceb1fdf"
URL = (f"https://raw.githubusercontent.com/LHRYANG/FSD/{OFFICIAL_COMMIT}/"
       "ngram_model/fsd.py")
CACHE = os.path.join(ROOT, "tests", ".official_cache", f"official_fsd_ngram_{OFFICIAL_COMMIT[:10]}.py")


def load_official():
    if not os.path.exists(CACHE):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        try:
            with urllib.request.urlopen(URL, timeout=20) as r:
                open(CACHE, "wb").write(r.read())
        except Exception as e:
            print(f"SKIP: could not fetch official code ({e})")
            return None
    spec = importlib.util.spec_from_file_location("official_fsd_ngram", CACHE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NGram


def ours_and_official(NGram, prompt, generated, n, beta, vocab):
    dec = FSDDecoder(n_max=n, beta=beta)
    dec.reset()
    dec.prime(prompt)
    off = NGram(list(prompt), n, vocab, beta)
    for t in generated:
        dec._register(t)
        off.update(t)
    query = list(prompt) + list(generated)
    return dec, off, query


def compare(NGram, prompt, generated, n=3, beta=0.9, vocab=12, tol=1e-9):
    dec, off, query = ours_and_official(NGram, prompt, generated, n, beta, vocab)
    cands = list(range(vocab))
    ref = off.penalize(query, cands)[cands]
    got = dec._anti_lm_scores(cands)
    err = float((ref - got).abs().max())
    assert err < tol, (prompt, generated, n, beta, err, ref.tolist(), got.tolist())
    return got


def test_constructed(NGram):
    # history shorter than N-1: both return zero penalty
    assert float(compare(NGram, [1], [], n=3).abs().sum()) == 0.0
    # every order fires: a repeated trigram context, weights ~ 0.90/0.09/0.01
    got = compare(NGram, [1, 2, 3] * 30 + [1, 2], [], n=3)
    assert 0.90 < float(got[3]) < 1.0, got[3]  # ~0.87 + 0.087 + 0.0033
    # a higher order is missing: candidate follows only the shorter context
    compare(NGram, [5, 6, 7, 8, 6, 9, 4], [6], n=3)
    # unigram-only mass (no beta on the unigram term)
    compare(NGram, [1, 2, 3, 4, 5, 6], [7, 8], n=3)
    # candidate never seen
    compare(NGram, [1, 2, 3, 1, 2], [], n=3)
    # other orders/betas
    for n, beta in [(2, 0.9), (4, 0.9), (3, 0.5), (3, 0.99)]:
        compare(NGram, [1, 2, 3, 1, 2, 3, 4, 1, 2, 3], [1], n=n, beta=beta)


def test_random(NGram, trials=300):
    rng = random.Random(0)
    for _ in range(trials):
        vocab = rng.choice([4, 8, 12])
        prompt = [rng.randrange(vocab) for _ in range(rng.randint(0, 6))]
        generated = [rng.randrange(vocab) for _ in range(rng.randint(0, 40))]
        n = rng.choice([2, 3, 4])
        beta = rng.choice([0.5, 0.9])
        if len(prompt) + len(generated) < n - 1:
            continue
        compare(NGram, prompt, generated, n=n, beta=beta, vocab=vocab)


if __name__ == "__main__":
    NGram = load_official()
    if NGram is None:
        sys.exit(0)
    test_constructed(NGram)
    test_random(NGram)
    print(f"OK: FSDDecoder._anti_lm_scores matches the official NGram.penalize "
          f"(commit {OFFICIAL_COMMIT[:10]}) on constructed prefixes and 300 random "
          f"histories (max err < 1e-9).")
