"""
test_multistep_hazard.py

Diagnostic (not the full dataset builder) for the multi-step commitment
hypothesis: single-token hazard showed NO differentiation between
"immediately risky" and "novel" candidates (0.027 vs 0.028, build_hazard_
dataset.py's per-candidate result). This tests whether the real mechanism is
SUSTAINED commitment to a repeating pattern over several steps, not any
single token choice.

Design
------
For a set of contexts, split candidates into "risky" (immediate_risk > 0 --
choosing this token right now would complete a repeated n-gram) and "novel"
(immediate_risk == 0). For each candidate and each commitment length L in a
small grid:

  1. Force the candidate as step 1.
  2. Force GREEDY (argmax) continuation for the next L-1 steps -- this
     simulates "if nothing disrupts the model's own strongest inclination
     for a few steps", the sharpest test of whether short-term commitment
     locks in a loop.
  3. Resume normal temperature-T sampling for the remaining (horizon-L)
     steps, so genuine escape is still possible if the pattern doesn't hold.
  4. Check the validated persistent-loop event over the full horizon.

Reports mean hazard for risky vs novel candidates AT EACH L, so we can see
directly whether a differentiation gap emerges as L grows, and if so, at
roughly what L -- informing whether a multi-step hazard estimator is worth
building at all, and if so, how far ahead it needs to force-commit.

This is deliberately SMALL and FAST: it answers a yes/no mechanism question
before committing to a full training-set collection run at this design.

Usage
-----
  python3 test_multistep_hazard.py --ckpt runs/cosine/best.pt --data_dir data_out \
      --samples_glob "runs/*/samples_cosine.txt" \
      --n_contexts 60 --k_rollouts 5 --horizon 150 --temperature 0.10 \
      --commitment_lengths 1 3 5 8 15
"""

import os, sys, json, argparse, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from src.utils import load_json
from src.model import CharTransformerLM
from sampling_eval import persistent_loop_onset
from build_hazard_dataset import (sample_contexts_from_files, select_candidates,
                                   exact_risk_signal)


@torch.no_grad()
def rollout_with_commitment(model, stoi, itos, ctx_ids, candidate, L, horizon,
                            temperature, k_rollouts, device, batch_size=256):
    """
    K rollouts of a single (context, candidate, L) triple: force `candidate`
    then (L-1) greedy steps, then (horizon-L) temperature-sampled steps.
    Returns the empirical hazard (fraction of the K rollouts that loop).
    """
    unk = stoi.get(" ", 0)
    block_size = model.block_size
    base = list(ctx_ids) + [candidate]

    hits = 0
    for start in range(0, k_rollouts, batch_size):
        n = min(batch_size, k_rollouts - start)
        idx = torch.tensor([base] * n, dtype=torch.long, device=device)

        for step in range(1, horizon):  # step 0 (the candidate) already placed
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            if step < L:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        gen_part_all = idx[:, len(ctx_ids):].tolist()
        for gen_part in gen_part_all:
            text = "".join(itos.get(str(int(t)), "?") for t in gen_part)
            onset = persistent_loop_onset(text)
            if onset >= 0:
                hits += 1

    return hits / k_rollouts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/cosine/best.pt")
    ap.add_argument("--data_dir", default="data_out")
    ap.add_argument("--samples_glob", nargs="*", default=["runs/*/samples_cosine.txt"])
    ap.add_argument("--strategy_filter", nargs="*",
                     default=["greedy", "temp_0.05", "temp_0.1", "rep_penalty"])
    ap.add_argument("--n_contexts", type=int, default=30,
                     help="Small and fast by design -- this is a mechanism "
                          "probe, not the final training-data collection. "
                          "NOTE: unlike build_hazard_dataset.py, rollouts here "
                          "are NOT batched across (context,candidate) pairs "
                          "(different L values need different sampling "
                          "policies mid-rollout, which is harder to vectorise "
                          "cleanly) -- kept small deliberately so this stays "
                          "fast despite the lost batching efficiency.")
    ap.add_argument("--n_risky_per_context", type=int, default=1)
    ap.add_argument("--n_novel_per_context", type=int, default=1)
    ap.add_argument("--k_rollouts", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.10)
    ap.add_argument("--commitment_lengths", nargs="*", type=int,
                     default=[1, 3, 5, 8, 15])
    ap.add_argument("--out", default="multistep_hazard_probe.json")
    args = ap.parse_args()

    device = (torch.device("mps") if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = CharTransformerLM(
        vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
        n_layer=cfg["n_layer"], n_embd=cfg["n_embd"],
        n_head=cfg["n_head"], dropout=0.0).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    vocab = load_json(os.path.join(args.data_dir, "vocab.json"))
    stoi, itos = vocab["stoi"], vocab["itos"]

    print(f"Collecting {args.n_contexts} contexts...")
    contexts = sample_contexts_from_files(args.samples_glob, itos, args.n_contexts,
                                           strategy_filter=args.strategy_filter)
    print(f"Got {len(contexts)} contexts.")
    if not contexts:
        print("No contexts found. Aborting.")
        return

    rng = random.Random(0)

    print("Selecting risky and novel candidates per context...")
    pairs = []  # (ctx_ids, candidate, is_risky)
    for ctx in contexts:
        ids = [stoi.get(c, stoi.get(" ", 0)) for c in ctx]
        cands, probs = select_candidates(model, stoi, itos, ids, device, n_top=12, n_random=8)
        risky, novel = [], []
        for c in cands:
            r = exact_risk_signal(ids + [c])
            (risky if r > 0 else novel).append(c)
        rng.shuffle(risky); rng.shuffle(novel)
        for c in risky[:args.n_risky_per_context]:
            pairs.append((ids, c, True))
        for c in novel[:args.n_novel_per_context]:
            pairs.append((ids, c, False))

    n_risky = sum(1 for _, _, r in pairs if r)
    n_novel = sum(1 for _, _, r in pairs if not r)
    print(f"Probing {len(pairs)} (context, candidate) pairs "
          f"({n_risky} risky, {n_novel} novel) x {len(args.commitment_lengths)} "
          f"commitment lengths x {args.k_rollouts} rollouts each")
    total_rollouts = len(pairs) * len(args.commitment_lengths) * args.k_rollouts
    print(f"Total rollouts: {total_rollouts} x {args.horizon} steps "
          f"(NOT batched across pairs in this diagnostic -- small by design)")

    results = {L: {"risky": [], "novel": []} for L in args.commitment_lengths}

    import time
    t0 = time.time()
    done = 0
    for pi, (ctx_ids, cand, is_risky) in enumerate(pairs):
        for L in args.commitment_lengths:
            hz = rollout_with_commitment(model, stoi, itos, ctx_ids, cand, L,
                                         args.horizon, args.temperature,
                                         args.k_rollouts, device)
            results[L]["risky" if is_risky else "novel"].append(hz)
        done += 1
        if done == 3:
            per_pair = (time.time() - t0) / done
            eta_min = per_pair * (len(pairs) - done) / 60
            print(f"  {done}/{len(pairs)} pairs done "
                  f"({per_pair:.1f}s/pair -> ETA ~{eta_min:.1f} more minutes)")
        elif done % 10 == 0:
            elapsed = (time.time() - t0) / 60
            print(f"  {done}/{len(pairs)} pairs done [{elapsed:.1f} min elapsed]")

    print(f"\n{'L (commitment steps)':>22} {'risky mean hazard':>19} "
          f"{'novel mean hazard':>19} {'gap':>8}")
    print("-" * 72)
    summary = []
    for L in args.commitment_lengths:
        r = results[L]["risky"]; n = results[L]["novel"]
        rm = float(np.mean(r)) if r else float("nan")
        nm = float(np.mean(n)) if n else float("nan")
        gap = rm - nm
        print(f"{L:>22} {rm:>19.3f} {nm:>19.3f} {gap:>8.3f}")
        summary.append({"L": L, "risky_mean": rm, "novel_mean": nm, "gap": gap,
                        "risky_vals": r, "novel_vals": n})

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {args.out}")

    best = max(summary, key=lambda s: s["gap"] if s["gap"] == s["gap"] else -1)
    print(f"\nLargest risky-vs-novel gap at L={best['L']}: {best['gap']:.3f}")
    if best["gap"] < 0.05:
        print("Even the best commitment length shows a negligible gap (<0.05).")
        print("This would support the conclusion that per-candidate hazard --")
        print("single-token OR short-commitment -- is not a useful signal for")
        print("this model, and the negative result from the single-token test")
        print("should stand as the final finding rather than being revised further.")
    else:
        print(f"A meaningful gap emerges by L={best['L']} -- multi-step commitment")
        print("does carry predictive signal. Worth building the full training set")
        print("at this commitment length next.")


if __name__ == "__main__":
    main()