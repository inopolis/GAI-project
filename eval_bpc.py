"""
eval_bpc.py — Correct full-book BPC evaluation with confidence intervals

Fixes the estimate_loss() bug in train.py which used only 50 random
overlapping batches (+-400K chars) instead of the full held-out book

Two modes:
  --mode full : non-overlapping windows tiling the entire book
  --mode bootstrap : full-book BPC + 95% CI via block bootstrap

Usage:
  python3 eval_bpc.py --ckpt runs/baseline/best.pt --split val
  python3 eval_bpc.py --ckpt runs/baseline/best.pt --split test
  python3 eval_bpc.py --ckpt runs/cosine/best.pt   --split test --mode bootstrap
  python3 eval_bpc.py --ckpt runs/baseline/best.pt runs/cosine/best.pt --split test
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


# Helpers

def load_model_and_vocab(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    model = CharTransformerLM(
        vocab_size  = cfg["vocab_size"],
        block_size  = cfg["block_size"],
        n_layer     = cfg["n_layer"],
        n_embd      = cfg["n_embd"],
        n_head      = cfg["n_head"],
        dropout     = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_split(data_dir, split, vocab_size):
    dtype = np.uint16 if vocab_size < 65535 else np.uint32
    path  = os.path.join(data_dir, f"{split}.bin")
    data  = np.fromfile(path, dtype=dtype).astype(np.int64)
    print(f"  Loaded {split}.bin: {len(data):,} characters")
    return data


#Full-book NLL over non-overlapping windows

@torch.no_grad()
def full_book_nll(model, data, block_size, batch_size=64, device="cpu"):
    """
    Tiles the entire token sequence with non-overlapping windows of length
    block_size, computes cross-entropy loss token-by-token, and returns
    the per-token NLL array (nats).

    Returns: np.ndarray of shape (N_tokens,) — per-token NLL in nats.
    """
    # Build non-overlapping (x, y) pairs
    # We need at least block_size+1 tokens to form one (x,y) pair.
    n = len(data)
    # Number of complete non-overlapping windows
    n_windows = (n - 1) // block_size
    if n_windows == 0:
        raise ValueError(f"Data too short ({n} tokens) for block_size={block_size}")

    # Collect per-token losses
    all_nlls = []

    for start in range(0, n_windows * block_size, block_size * batch_size):
        batch_xs, batch_ys = [], []
        for b in range(batch_size):
            s = start + b * block_size
            if s + block_size + 1 > n:
                break
            x = data[s : s + block_size]
            y = data[s + 1 : s + block_size + 1]
            batch_xs.append(x)
            batch_ys.append(y)

        if not batch_xs:
            break

        x_t = torch.tensor(np.stack(batch_xs), dtype=torch.long, device=device)
        y_t = torch.tensor(np.stack(batch_ys), dtype=torch.long, device=device)

        logits, _ = model(x_t)  # (B, T, V)
        B, T, V = logits.shape

        # Per-token NLL (nats)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (B, T, V)
        token_nll = -log_probs.gather(
            2, y_t.unsqueeze(-1)
        ).squeeze(-1)  # (B, T)

        all_nlls.append(token_nll.cpu().numpy().reshape(-1))

    return np.concatenate(all_nlls)  # (N_tokens,)


#Block bootstrap CI

def block_bootstrap_bpc(nll_tokens, n_boot=1000, block_len=512, seed=0):
    """
    Block bootstrap for 95% CI on mean BPC.
    Preserves local autocorrelation structure by sampling contiguous blocks.

    Returns: (mean_bpc, ci_low_95, ci_high_95, std_bpc)
    """
    rng = np.random.default_rng(seed)
    n   = len(nll_tokens)
    n_blocks = max(1, n // block_len)
    boot_means = []

    for _ in range(n_boot):
        starts   = rng.integers(0, n - block_len + 1, size=n_blocks)
        sample   = np.concatenate([nll_tokens[s : s + block_len] for s in starts])
        boot_means.append(sample.mean())

    boot_means = np.array(boot_means) / math.log(2)  # nats → bits
    mean_bpc   = nll_tokens.mean() / math.log(2)
    ci_low     = np.percentile(boot_means, 2.5)
    ci_high    = np.percentile(boot_means, 97.5)
    return float(mean_bpc), float(ci_low), float(ci_high), float(boot_means.std())


# Main

def evaluate_checkpoint(ckpt_path, data_dir, split, mode, batch_size, device):
    print(f"\n{'='*60}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Split      : {split}   Mode: {mode}")
    print(f"{'='*60}")

    model, cfg = load_model_and_vocab(ckpt_path, device)
    block_size  = cfg["block_size"]
    vocab_size  = cfg["vocab_size"]

    data = load_split(data_dir, split, vocab_size)

    t0 = time.time()
    nll_tokens = full_book_nll(model, data, block_size, batch_size=batch_size, device=device)
    elapsed = time.time() - t0

    n_tokens  = len(nll_tokens)
    mean_nll  = float(nll_tokens.mean()) # nats
    mean_bpc  = mean_nll / math.log(2) # bits

    result = {
        "ckpt"        : ckpt_path,
        "split"       : split,
        "mode"        : mode,
        "n_tokens"    : int(n_tokens),
        "mean_nll_nats": round(mean_nll, 6),
        "mean_bpc"    : round(mean_bpc, 4),
        "elapsed_sec" : round(elapsed, 1),
    }

    if mode == "bootstrap":
        print("  Running block bootstrap (n=1000)")
        mean_bpc_b, ci_low, ci_high, std_b = block_bootstrap_bpc(nll_tokens)
        result["ci95_low"]  = round(ci_low,  4)
        result["ci95_high"] = round(ci_high, 4)
        result["std_bpc"]   = round(std_b,   4)
        print(f"  BPC = {mean_bpc_b:.4f}  95% CI [{ci_low:.4f}, {ci_high:.4f}]  ±{std_b:.4f}")
    else:
        print(f"  BPC = {mean_bpc:.4f}  ({n_tokens:,} tokens in {elapsed:.1f}s)")

    return result


def main():
    ap = argparse.ArgumentParser(description="Correct full-book BPC evaluation")
    ap.add_argument("--ckpt",       type=str, nargs="+", required=True,
                    help="One or more checkpoint paths")
    ap.add_argument("--data_dir",   type=str, default="data_out")
    ap.add_argument("--split",      type=str, default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--mode",       type=str, default="bootstrap",
                    choices=["full", "bootstrap"],
                    help="full: non-overlapping windows only; bootstrap: + 95% CI")
    ap.add_argument("--batch_size", type=int, default=64,
                    help="Batch size for forward passes (reduce if OOM)")
    ap.add_argument("--out",        type=str, default=None,
                    help="Optional JSON output path")
    args = ap.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    results = []
    for ckpt in args.ckpt:
        r = evaluate_checkpoint(
            ckpt_path  = ckpt,
            data_dir   = args.data_dir,
            split      = args.split,
            mode       = args.mode,
            batch_size = args.batch_size,
            device     = device,
        )
        results.append(r)

    # Summary table 
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  COMPARISON SUMMARY")
        print(f"{'='*60}")
        header = f"  {'Checkpoint':<35} {'BPC':>7}"
        if args.mode == "bootstrap":
            header += f"  {'CI95':>18}  {'std':>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in results:
            name = os.path.basename(os.path.dirname(r["ckpt"]))
            line = f"  {name:<35} {r['mean_bpc']:>7.4f}"
            if args.mode == "bootstrap":
                ci = f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]"
                line += f"  {ci:>18}  {r['std_bpc']:>6.4f}"
            print(line)

        # Check overlap of CIs (if bootstrap)
        if args.mode == "bootstrap" and len(results) == 2:
            r0, r1 = results[0], results[1]
            overlap = r0["ci95_low"] <= r1["ci95_high"] and r1["ci95_low"] <= r0["ci95_high"]
            print(f"\n  CI overlap: {'YES — difference NOT significant at 95%' if overlap else 'NO — difference IS significant at 95%'}")

    #Save
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved → {args.out}")
    else:
        # Default: save next to first checkpoint
        out_dir = os.path.dirname(args.ckpt[0])
        out_path = os.path.join(out_dir, f"eval_bpc_{args.split}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
