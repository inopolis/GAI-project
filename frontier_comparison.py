"""
frontier_comparison.py

Real-model test of the prediction of Section 4 of the paper: at equal loop rate,
a rule that fixes the PRICE of risk (one multiplier lambda for the whole
generation; recurrence-risk "fixed" mode) should cost less KL than a rule that
fixes a per-step risk BUDGET (a constant epsilon re-solved at every step;
recurrence-risk "dual" mode), because a fixed budget implies per-step
multipliers that vary widely.

The script sweeps both families on the primary character-level model, with the
same prompts, seeds, risk definition (n-gram orders 3-6), detector and
KL accounting as common_kl_comparison.py (it imports that module's generation
loop), and records for every generation its loop event, path KL, and the
per-step multiplier trace of the decoder (decoder.alpha_history), so the
dispersion of the realised multipliers can be reported next to the KL.

Output: <out_dir>/frontier_rows.csv      one row per configuration
        <out_dir>/frontier_samples.csv   one row per generation

Usage (full run, 150 generations per configuration):
  python3 frontier_comparison.py --ckpt runs/cosine/best.pt --data_dir data_out \
      --out_dir runs/frontier_v1 --n_seeds 10 --n_chars 500
Quick timing check:
  python3 frontier_comparison.py --out_dir /tmp/frontier_smoke --n_seeds 1 --n_prompts 2 \
      --only rr_fixed_lam2 rr_dual_eps0.05
"""
import os, sys, argparse, csv, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from src.utils import set_seed, load_json, ensure_dir
from src.decoding import RecurrenceRiskDecoder
from sampling_eval import (TEST_PROMPTS, encode, decode, load_model, cluster_bootstrap_ci,
                           persistent_loop_onset, survival_auc, rmst, loop_rate)
from common_kl_comparison import generate_and_score, TemperatureBaseline

FIXED_LAMBDAS = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
DUAL_EPS = [0.65, 0.60, 0.50, 0.30, 0.20, 0.10, 0.05, 0.01]


def make_configs():
    cfg = {"temperature_0.10": lambda: TemperatureBaseline(temperature=0.10)}
    for lam in FIXED_LAMBDAS:
        cfg[f"rr_fixed_lam{lam:g}"] = (lambda lam=lam: RecurrenceRiskDecoder(
            temperature=0.10, top_p=1.0, n_min=3, n_max=6, mode="fixed", alpha_base=lam))
    for eps in DUAL_EPS:
        cfg[f"rr_dual_eps{eps:g}"] = (lambda eps=eps: RecurrenceRiskDecoder(
            temperature=0.10, top_p=1.0, n_min=3, n_max=6, mode="dual", eps=eps))
    return cfg


def lambda_stats(decoder):
    lam = np.asarray(getattr(decoder, "alpha_history", []), dtype=float)
    if lam.size == 0:
        return dict(lam_mean=float("nan"), lam_std=float("nan"), lam_frac_zero=float("nan"),
                    lam_p90=float("nan"), lam_max=float("nan"))
    return dict(lam_mean=float(lam.mean()), lam_std=float(lam.std()),
                lam_frac_zero=float((lam <= 1e-9).mean()),
                lam_p90=float(np.quantile(lam, 0.9)), lam_max=float(lam.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/cosine/best.pt")
    ap.add_argument("--data_dir", default="data_out")
    ap.add_argument("--out_dir", default="runs/frontier_v1")
    ap.add_argument("--n_chars", type=int, default=500)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--n_prompts", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    prompts = TEST_PROMPTS if args.n_prompts is None else TEST_PROMPTS[:args.n_prompts]
    device = (torch.device("mps") if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Device: {device}  Prompts: {len(prompts)}  Seeds: {args.n_seeds}")
    model, _ = load_model(args.ckpt, device)
    vocab = load_json(os.path.join(args.data_dir, "vocab.json"))
    stoi, itos = vocab["stoi"], vocab["itos"]
    ensure_dir(args.out_dir)

    configs = make_configs()
    if args.only:
        unknown = set(args.only) - set(configs)
        if unknown:
            raise SystemExit(f"unknown configuration(s) in --only: {sorted(unknown)}; known: {sorted(configs)}")
        configs = {k: v for k, v in configs.items() if k in set(args.only)}

    rows, samples = [], []
    for name, factory in configs.items():
        t0 = time.time()
        kl_all, onsets, groups, lam_rows = [], [], [], []
        for pi, (pname, ptext) in enumerate(prompts):
            for seed in range(1, args.n_seeds + 1):
                set_seed(seed)
                prompt_ids = encode(ptext, stoi)[0].tolist()
                decoder = factory()
                shadow = RecurrenceRiskDecoder(n_min=3, n_max=6, mode="fixed", alpha_base=0.0)
                gen_ids, kls, risks, _ = generate_and_score(
                    model, prompt_ids, args.n_chars, device, decoder, shadow)
                text = decode(gen_ids[len(prompt_ids):], itos)
                onset = persistent_loop_onset(text)
                ls = lambda_stats(decoder)
                kl_all.append(float(np.mean(kls)))
                onsets.append(onset)
                groups.append(pi)
                lam_rows.append(ls)
                samples.append({"strategy": name, "prompt_index": pi, "seed": seed,
                                "loop_onset": onset, "loop_event": int(onset >= 0),
                                "kl_bits_per_step_mean": kl_all[-1],
                                "kl_bits_total": float(np.sum(kls)),
                                "n_steps": len(kls), "risk_achieved_mean": float(np.mean(risks)),
                                **ls})
                print(f"\r  {name:<22} {len(kl_all)}/{len(prompts)*args.n_seeds}", end="", flush=True)
        m, lo, hi = cluster_bootstrap_ci(kl_all, groups)
        row = {"strategy": name, "n_samples": len(kl_all), "kl_bits_per_step_mean": m,
               "kl_lo": lo, "kl_hi": hi, "kl_bits_total_mean": m * args.n_chars,
               "loop_rate": loop_rate(onsets), "loop_count": int(sum(o >= 0 for o in onsets)),
               "rmst": rmst(onsets, args.n_chars), "seconds": time.time() - t0}
        for k in ["lam_mean", "lam_std", "lam_frac_zero", "lam_p90", "lam_max"]:
            row[k] = float(np.nanmean([r[k] for r in lam_rows]))
        rows.append(row)
        print(f"\r  {name:<22} KL/step={m:.4f}  loops={row['loop_count']}/{len(kl_all)}  "
              f"lam mean/std={row['lam_mean']:.2f}/{row['lam_std']:.2f}  "
              f"frac(lam=0)={row['lam_frac_zero']:.2f}  [{row['seconds']:.0f}s]")
        with open(os.path.join(args.out_dir, "frontier_rows.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        with open(os.path.join(args.out_dir, "frontier_samples.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
            w.writeheader(); w.writerows(samples)
    print(f"\nWrote {args.out_dir}/frontier_rows.csv and frontier_samples.csv")


if __name__ == "__main__":
    main()
