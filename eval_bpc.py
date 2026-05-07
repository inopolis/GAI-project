"""
eval_bpc.py — Full-book BPC evaluation with paired bootstrap significance test.

Modes:
  --mode full        : non-overlapping windows over the entire book, no CI
  --mode bootstrap   : full-book BPC + 95% CI via block bootstrap (single model)
  --mode paired      : paired bootstrap test comparing TWO checkpoints
                       (requires exactly 2 --ckpt arguments)

Usage examples:
  # Single model, full-book eval
  python3 eval_bpc.py --ckpt runs/baseline/best.pt --split test --mode full

  # Single model with 95% CI
  python3 eval_bpc.py --ckpt runs/baseline/best.pt --split test --mode bootstrap

  # Paired bootstrap: baseline vs cosine
  python3 eval_bpc.py \
      --ckpt runs/baseline/best.pt runs/cosine/best.pt \
      --split test --mode paired

NOTE ON OLD RESULTS (outdated — kept for reference only):
  The original estimate_loss() in train.py sampled only 50 random overlapping
  batches, covering ~27 percent of the test book. This inflated BPC by ~1.46 bits:
    Baseline test BPC (original, INCORRECT): 3.702
    Cosine   test BPC (original, INCORRECT): 3.530
  The corrected full-book values are ~2.238 and ~2.234 respectively.
  Do not cite the old numbers.
"""

import os
import sys
import argparse
import json
import math
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import load_json, bpc_from_loss
from src.model import CharTransformerLM


# Model loading

def load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
    model = CharTransformerLM(
        vocab_size = cfg["vocab_size"],
        block_size = cfg["block_size"],
        n_layer    = cfg["n_layer"],
        n_embd     = cfg["n_embd"],
        n_head     = cfg["n_head"],
        dropout    = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_split(data_dir, split, vocab_size):
    dtype = np.uint16 if vocab_size < 65535 else np.uint32
    path  = os.path.join(data_dir, f"{split}.bin")
    data  = np.fromfile(path, dtype=dtype).astype(np.int64)
    print(f"  Loaded {split}.bin — {len(data):,} characters")
    return data


# Per-token NLL over non-overlapping windows

@torch.no_grad()
def full_book_nll(model, data, block_size, batch_size=64, device="cpu"):
    """
    Tiles the entire sequence with non-overlapping windows of block_size.
    Returns per-token NLL array (nats), shape (N_tokens,).
    """
    n     = len(data)
    n_win = (n - 1) // block_size
    if n_win == 0:
        raise ValueError(f"Data too short ({n}) for block_size={block_size}")

    all_nll = []
    for start in range(0, n_win * block_size, block_size * batch_size):
        xs, ys = [], []
        for b in range(batch_size):
            s = start + b * block_size
            if s + block_size + 1 > n:
                break
            xs.append(data[s : s + block_size])
            ys.append(data[s + 1 : s + block_size + 1])
        if not xs:
            break
        x_t = torch.tensor(np.stack(xs), dtype=torch.long, device=device)
        y_t = torch.tensor(np.stack(ys), dtype=torch.long, device=device)
        logits, _ = model(x_t)
        log_p     = torch.nn.functional.log_softmax(logits, dim=-1)
        tok_nll   = -log_p.gather(2, y_t.unsqueeze(-1)).squeeze(-1)
        all_nll.append(tok_nll.cpu().numpy().reshape(-1))

    return np.concatenate(all_nll)


# Single-model block bootstrap

def block_bootstrap_bpc(nll, n_boot=2000, block_len=512, seed=0):
    """Block bootstrap 95% CI for mean BPC."""
    rng      = np.random.default_rng(seed)
    n        = len(nll)
    n_blocks = max(1, n // block_len)
    boots    = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        sample = np.concatenate([nll[s : s + block_len] for s in starts])
        boots.append(sample.mean())
    boots    = np.array(boots) / math.log(2)
    mean_bpc = float(nll.mean() / math.log(2))
    return (mean_bpc,
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)),
            float(boots.std()))


# Paired bootstrap test

def paired_bootstrap_test(nll_a, nll_b, n_boot=2000, block_len=512, seed=0):
    """
    Paired block bootstrap significance test for H0: E[NLL_A] == E[NLL_B].

    Uses per-token NLL differences delta[i] = NLL_A[i] - NLL_B[i].
    Positive delta_bpc means model A is worse (higher loss) than model B.

    The paired design is stronger than comparing two independent CIs:
    it cancels out position-level variance that is common to both models,
    so the test has higher power to detect genuine differences.

    Two-sided p-value: fraction of bootstrap deltas at least as extreme
    as the observed delta, after centring the bootstrap distribution at 0.
    """
    n     = min(len(nll_a), len(nll_b))
    delta = nll_a[:n] - nll_b[:n]         # per-token difference (nats)
    observed_bpc = delta.mean() / math.log(2)

    rng      = np.random.default_rng(seed)
    n_blocks = max(1, n // block_len)
    boot_deltas = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        sample = np.concatenate([delta[s : s + block_len] for s in starts])
        boot_deltas.append(sample.mean() / math.log(2))

    boot_deltas  = np.array(boot_deltas)
    boot_centred = boot_deltas - boot_deltas.mean()   # shift to H0: mean=0
    p_value      = float(np.mean(np.abs(boot_centred) >= abs(observed_bpc)))

    return {
        "delta_bpc_A_minus_B" : round(float(observed_bpc), 5),
        "ci95_low"            : round(float(np.percentile(boot_deltas, 2.5)),  5),
        "ci95_high"           : round(float(np.percentile(boot_deltas, 97.5)), 5),
        "p_value"             : round(p_value, 4),
        "significant_p05"     : bool(p_value < 0.05),
        "n_boot"              : n_boot,
        "block_len"           : block_len,
        "n_tokens_compared"   : int(n),
    }


# Per-model evaluation 

def evaluate_one(ckpt_path, data, block_size, batch_size, device, mode):
    model, cfg = load_model(ckpt_path, device)
    t0         = time.time()
    nll        = full_book_nll(model, data, block_size, batch_size, device)
    elapsed    = time.time() - t0
    del model

    mean_bpc = float(nll.mean() / math.log(2))
    result   = {
        "ckpt"        : ckpt_path,
        "n_tokens"    : int(len(nll)),
        "mean_bpc"    : round(mean_bpc, 4),
        "elapsed_sec" : round(elapsed, 1),
    }
    if mode in ("bootstrap", "paired"):
        m, lo, hi, sd = block_bootstrap_bpc(nll)
        result["ci95_low"]  = round(lo, 4)
        result["ci95_high"] = round(hi, 4)
        result["std_bpc"]   = round(sd, 4)

    return result, nll


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       nargs="+", required=True)
    ap.add_argument("--data_dir",   default="data_out")
    ap.add_argument("--split",      default="test", choices=["train","val","test"])
    ap.add_argument("--mode",       default="bootstrap",
                    choices=["full","bootstrap","paired"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_boot",     type=int, default=2000)
    ap.add_argument("--out",        default=None)
    args = ap.parse_args()

    if args.mode == "paired" and len(args.ckpt) != 2:
        ap.error("--mode paired requires exactly 2 --ckpt arguments")

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device : {device}")
    print(f"Mode   : {args.mode}   Split: {args.split}   n_boot: {args.n_boot}\n")

    # Load split data once — shared across both models
    _, tmp_cfg = load_model(args.ckpt[0], device)
    vocab_size = tmp_cfg["vocab_size"]
    block_size = tmp_cfg["block_size"]
    data       = load_split(args.data_dir, args.split, vocab_size)

    results  = []
    nll_list = []

    for ckpt in args.ckpt:
        name = os.path.basename(os.path.dirname(ckpt))
        print(f"{'─'*58}")
        print(f"  {name}  ({ckpt})")
        print(f"{'─'*58}")
        r, nll = evaluate_one(ckpt, data, block_size,
                              args.batch_size, device, args.mode)
        nll_list.append(nll)
        if "ci95_low" in r:
            print(f"  BPC={r['mean_bpc']:.4f}  "
                  f"95%CI=[{r['ci95_low']:.4f},{r['ci95_high']:.4f}]  "
                  f"±{r['std_bpc']:.4f}  ({r['elapsed_sec']:.1f}s)")
        else:
            print(f"  BPC={r['mean_bpc']:.4f}  "
                  f"({r['n_tokens']:,} tokens, {r['elapsed_sec']:.1f}s)")
        results.append(r)

    # ── Paired bootstrap test ─────────────────────────────────────────────────
    paired_result = None
    if args.mode == "paired":
        print(f"\n{'═'*58}")
        print("  PAIRED BOOTSTRAP TEST")
        print(f"  A = {os.path.basename(os.path.dirname(args.ckpt[0]))}")
        print(f"  B = {os.path.basename(os.path.dirname(args.ckpt[1]))}")
        print(f"{'═'*58}")
        paired_result = paired_bootstrap_test(
            nll_list[0], nll_list[1], n_boot=args.n_boot
        )
        pr = paired_result
        print(f"  Observed delta BPC (A - B) : {pr['delta_bpc_A_minus_B']:+.5f}")
        print(f"  Bootstrap 95% CI           : "
              f"[{pr['ci95_low']:+.5f}, {pr['ci95_high']:+.5f}]")
        print(f"  Two-sided p-value          : {pr['p_value']:.4f}")
        if pr["significant_p05"]:
            better = "B" if pr["delta_bpc_A_minus_B"] > 0 else "A"
            print(f"  => SIGNIFICANT (p < 0.05) — model {better} is reliably better")
        else:
            print(f"  => NOT significant (p >= 0.05) — no reliable difference")
        print(f"  (n_boot={pr['n_boot']}, block_len={pr['block_len']}, "
              f"n_tokens={pr['n_tokens_compared']:,})")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {"results": results}
    if paired_result:
        output["paired_test"] = paired_result

    out_path = args.out or os.path.join(
        os.path.dirname(args.ckpt[0]),
        f"eval_bpc_{args.split}_{args.mode}.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()