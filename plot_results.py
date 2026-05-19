"""
plot_results.py

Generates all plots from sampling evaluation results:
  1. Pareto plots: survival AUC vs NLL, vs n-gram similarity, vs spelling error.
     Hard no-repeat shown separately because it directly forbids the measured event.
  2. Kaplan-Meier survival curves for key methods.
  3. RMST comparison bar chart with confidence intervals.
  4. Ablation bar charts for survival AUC, NLL, n-gram similarity.
  5. Runtime overhead (chars/sec) bar chart.

Usage:
  python3 plot_results.py \
    --pareto runs/sampling_eval_v5/pareto_data.csv \
    --km     runs/sampling_eval_v5/survival_curves.json \
    --full   runs/sampling_eval_v5/all_results.json \
    --out    runs/sampling_eval_v5/plots
"""

import os, sys, json, argparse, csv
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


COLORS = {
    "baseline":        "#6B7280",
    "baseline_sweep":  "#9CA3AF",
    "probabilistic":   "#3B82F6",
    "strong_baseline": "#F59E0B",
    "hard_constraint": "#EF4444",
    "ablation":        "#8B5CF6",
    "adaptive":        "#10B981",
}
MARKER = {
    "baseline": "o", "baseline_sweep": "o", "probabilistic": "s",
    "strong_baseline": "D", "hard_constraint": "X",
    "ablation": "^", "adaptive": "*",
}
LABEL = {
    "adaptive_full": "Adaptive (ours)", "ablation_risk_only": "Risk-only",
    "ablation_entropy_only": "Entropy-only", "ablation_fixed_alpha": "Fixed-alpha",
    "ablation_no_top_p": "No top-p", "ablation_narrow_ngram": "Narrow n-gram",
    "ablation_wide_ngram": "Wide n-gram", "ablation_hard_in_adaptive": "Hard+adaptive",
    "rep_penalty_1.3": "Rep.penalty 1.3", "mirostat_tau5": "Mirostat τ=5",
    "temp_0.8": "Temp 0.8", "nucleus_p0.95": "Nucleus p=0.95",
    "no_repeat_4gram": "No-repeat 4-gram*", "lz_decoder": "LZ-decoder",
    "greedy": "Greedy", "typical_p0.9": "Typical p=0.9",
}


def load_pareto(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                if v in ("", "None"):
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows


def load_json_file(path):
    with open(path) as f:
        return json.load(f)


def pareto_plot(rows, x_key, y_key, x_label, y_label, title, out_path, checkpoint=None):
    if checkpoint:
        rows = [r for r in rows if r.get("checkpoint") == checkpoint]

    soft = [r for r in rows
            if str(r.get("hard_constraint")) != "True"
            and isinstance(r.get(x_key), float)
            and isinstance(r.get(y_key), float)]
    hard = [r for r in rows
            if str(r.get("hard_constraint")) == "True"
            and isinstance(r.get(x_key), float)
            and isinstance(r.get(y_key), float)]

    def on_frontier(r, pool):
        return not any(
            q[x_key] >= r[x_key] and q[y_key] >= r[y_key]
            and (q[x_key] > r[x_key] or q[y_key] > r[y_key])
            for q in pool if q is not r
        )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    labelled = set()

    for r in soft:
        cat   = r.get("category", "baseline")
        c     = COLORS.get(cat, "#6B7280")
        m     = MARKER.get(cat, "o")
        front = on_frontier(r, soft)
        ax.scatter(r[x_key], r[y_key], c=c, marker=m,
                   s=120 if front else 45,
                   alpha=1.0 if front else 0.4,
                   edgecolors="white", linewidths=1.5 if front else 0.5, zorder=3)
        name = r.get("strategy", "")
        if front and name not in labelled:
            ax.annotate(LABEL.get(name, name), (r[x_key], r[y_key]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=7.5, color=c)
            labelled.add(name)

    for r in hard:
        c    = COLORS["hard_constraint"]
        name = r.get("strategy", "")
        ax.scatter(r[x_key], r[y_key], c=c, marker="X",
                   s=150, alpha=0.9, edgecolors="white", linewidths=1.5, zorder=4)
        ax.annotate(LABEL.get(name, name), (r[x_key], r[y_key]),
                    textcoords="offset points", xytext=(6, -11),
                    fontsize=7.5, color=c)

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    legend_items = [mpatches.Patch(color=v, label=k.replace("_", " ").title())
                    for k, v in COLORS.items()]
    ax.legend(handles=legend_items, fontsize=7.5, loc="lower right", framealpha=0.7)
    fig.text(0.01, 0.01,
             "* Hard no-repeat directly forbids the measured event — not comparable.",
             fontsize=7, color="#6B7280")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def km_plot(km_data, checkpoint, strategies, out_path, n_chars=500):
    ckpt_data = km_data.get(checkpoint, {})
    fig, ax   = plt.subplots(figsize=(8, 5))

    for strat in strategies:
        d = ckpt_data.get(strat)
        if not d:
            continue
        ts  = d["km_times"]
        S   = d["km_survival"]
        cat = ("adaptive"        if "adaptive" in strat else
               "hard_constraint" if "no_repeat" in strat else
               "probabilistic"   if strat in ("rep_penalty_1.3","mirostat_tau5","lz_decoder") else
               "baseline")
        color = COLORS.get(cat, "#6B7280")
        lw    = 2.5 if "adaptive" in strat else 1.8
        ls    = "--" if "no_repeat" in strat else "-"
        label = LABEL.get(strat, strat)
        plot_t = [0] + ts + [n_chars]
        plot_s = [1.0] + S + ([S[-1]] if S else [1.0])
        ax.step(plot_t, plot_s, where="post",
                color=color, linewidth=lw, linestyle=ls, label=label)

    ax.set_xlabel("Characters generated", fontsize=11)
    ax.set_ylabel("Fraction loop-free (survival)", fontsize=11)
    ax.set_title(f"Loop-onset survival curves  [{checkpoint}]", fontsize=12)
    ax.set_xlim(0, n_chars)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.8)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    fig.text(0.01, 0.01,
             "Dashed = hard constraint. Censored samples (no loop) counted throughout.",
             fontsize=7, color="#6B7280")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def rmst_bar(all_results, checkpoint, out_path):
    key = {
        "greedy","temp_0.8","nucleus_p0.95","rep_penalty_1.3",
        "mirostat_tau5","lz_decoder","no_repeat_4gram",
        "adaptive_full","ablation_risk_only","ablation_entropy_only","ablation_fixed_alpha",
    }
    rows = [r for r in all_results.get(checkpoint, []) if r["strategy"] in key]
    rows.sort(key=lambda r: r.get("rmst", 0))
    if not rows:
        return

    names  = [LABEL.get(r["strategy"], r["strategy"]) for r in rows]
    vals   = [r.get("rmst", 0) for r in rows]
    lo_err = [max(0, r.get("rmst", 0) - r.get("rmst_ci_lo", r.get("rmst", 0))) for r in rows]
    hi_err = [max(0, r.get("rmst_ci_hi", r.get("rmst", 0)) - r.get("rmst", 0)) for r in rows]
    colors = [COLORS.get(r.get("category","baseline"), "#6B7280") for r in rows]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names)*0.5)))
    y_pos   = np.arange(len(names))
    ax.barh(y_pos, vals, xerr=[lo_err, hi_err], color=colors,
            capsize=4, alpha=0.85, error_kw={"elinewidth":1.2, "ecolor":"#374151"})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("RMST — expected loop-free characters (τ=500)", fontsize=10)
    ax.set_title(f"Restricted Mean Survival Time  [{checkpoint}]", fontsize=12)
    ax.axvline(500, color="#E5E7EB", linewidth=0.8, linestyle="--")
    ax.grid(True, axis="x", alpha=0.2, linewidth=0.5)
    legend_items = [mpatches.Patch(color=v, label=k.replace("_"," ").title())
                    for k, v in COLORS.items()]
    ax.legend(handles=legend_items, fontsize=7.5, loc="lower right", framealpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def runtime_bar(all_results, checkpoint, out_path):
    rows = [r for r in all_results.get(checkpoint, [])
            if r.get("chars_per_sec") is not None]
    if not rows:
        print("  No runtime data available — skipping")
        return
    rows.sort(key=lambda r: r.get("chars_per_sec", 0), reverse=True)
    names  = [LABEL.get(r["strategy"], r["strategy"]) for r in rows]
    vals   = [r["chars_per_sec"] for r in rows]
    colors = [COLORS.get(r.get("category","baseline"), "#6B7280") for r in rows]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names)*0.45)))
    y_pos   = np.arange(len(names))
    ax.barh(y_pos, vals, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Throughput (chars / sec)", fontsize=10)
    ax.set_title(f"Runtime overhead  [{checkpoint}]", fontsize=12)
    ax.grid(True, axis="x", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def ablation_bar(all_results, checkpoint, metric, ylabel, title_suffix, out_path,
                 higher_better=True):
    ablation_names = {
        "adaptive_full","ablation_fixed_alpha","ablation_risk_only",
        "ablation_entropy_only","ablation_no_top_p","ablation_narrow_ngram",
        "ablation_wide_ngram","ablation_hard_in_adaptive",
    }
    rows = [r for r in all_results.get(checkpoint, [])
            if r["strategy"] in ablation_names]
    rows.sort(key=lambda r: r.get(f"{metric}_mean", 0), reverse=higher_better)
    if not rows:
        return

    names  = [LABEL.get(r["strategy"], r["strategy"]) for r in rows]
    vals   = [r.get(f"{metric}_mean", 0) for r in rows]
    lo_err = [max(0, r.get(f"{metric}_mean", 0) - r.get(f"{metric}_ci_lo", r.get(f"{metric}_mean", 0)))
              for r in rows]
    hi_err = [max(0, r.get(f"{metric}_ci_hi", r.get(f"{metric}_mean", 0)) - r.get(f"{metric}_mean", 0))
              for r in rows]
    colors = ["#10B981" if r["strategy"] == "adaptive_full" else "#8B5CF6" for r in rows]

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(names)*0.55)))
    y_pos   = np.arange(len(names))
    ax.barh(y_pos, vals, xerr=[lo_err, hi_err], color=colors,
            capsize=4, alpha=0.85, error_kw={"elinewidth":1.2, "ecolor":"#374151"})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(ylabel, fontsize=10)
    ax.set_title(f"Ablation: {title_suffix}  [{checkpoint}]", fontsize=11)
    ax.grid(True, axis="x", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pareto", default="runs/sampling_eval_v5/pareto_data.csv")
    ap.add_argument("--km",     default="runs/sampling_eval_v5/survival_curves.json")
    ap.add_argument("--full",   default="runs/sampling_eval_v5/all_results.json")
    ap.add_argument("--out",    default="runs/sampling_eval_v5/plots")
    ap.add_argument("--n_chars",type=int, default=500)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pareto_rows = load_pareto(args.pareto)
    km_data     = load_json_file(args.km) if args.km and os.path.exists(args.km) else {}
    all_results = load_json_file(args.full)
    checkpoints = list(all_results.keys())

    key_methods = [
        "greedy","temp_0.8","nucleus_p0.95","rep_penalty_1.3",
        "mirostat_tau5","lz_decoder","no_repeat_4gram",
        "adaptive_full","ablation_risk_only","ablation_entropy_only",
    ]

    for ckpt in checkpoints:
        tag = ckpt.replace("/","_")
        print(f"\n  Plots for: {ckpt}")

        pareto_plot(pareto_rows, "survival_auc", "gen_nll_bpc",
                    "Survival AUC (↑ fewer loops)",
                    "Generated-text NLL BPC (↓ higher quality)",
                    "Pareto: Survival vs Quality (NLL)",
                    os.path.join(args.out, f"pareto_survival_nll_{tag}.png"),
                    checkpoint=ckpt)

        pareto_plot(pareto_rows, "survival_auc", "ngram_sim_4",
                    "Survival AUC (↑ fewer loops)",
                    "4-gram similarity to real text (↑ more natural)",
                    "Pareto: Survival vs Distributional Similarity",
                    os.path.join(args.out, f"pareto_survival_sim_{tag}.png"),
                    checkpoint=ckpt)

        pareto_plot(pareto_rows, "survival_auc", "spelling_error",
                    "Survival AUC (↑ fewer loops)",
                    "Spelling error rate (↓ fewer invented words)",
                    "Pareto: Survival vs Spelling",
                    os.path.join(args.out, f"pareto_survival_spelling_{tag}.png"),
                    checkpoint=ckpt)

        if km_data:
            km_plot(km_data, ckpt, key_methods,
                    os.path.join(args.out, f"km_survival_{tag}.png"),
                    n_chars=args.n_chars)
        else:
            print("  Skipping KM plot — survival_curves.json not found")

        rmst_bar(all_results, ckpt,
                 os.path.join(args.out, f"rmst_bar_{tag}.png"))

        runtime_bar(all_results, ckpt,
                    os.path.join(args.out, f"runtime_{tag}.png"))

        ablation_bar(all_results, ckpt, "survival_auc",
                     "Survival AUC", "Survival AUC",
                     os.path.join(args.out, f"ablation_sauc_{tag}.png"),
                     higher_better=True)

        ablation_bar(all_results, ckpt, "gen_nll_bpc",
                     "Generated-text NLL (BPC)", "Generated NLL",
                     os.path.join(args.out, f"ablation_nll_{tag}.png"),
                     higher_better=False)

        ablation_bar(all_results, ckpt, "ngram_sim_4",
                     "4-gram similarity to real text", "N-gram similarity",
                     os.path.join(args.out, f"ablation_sim_{tag}.png"),
                     higher_better=True)

    print(f"\n  All plots saved to {args.out}/")


if __name__ == "__main__":
    main()