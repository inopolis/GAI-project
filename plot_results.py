"""
plot_results.py — figures for Recurrence-Risk Decoding.

Reads the v6 evaluation outputs and produces:
  1. Pareto plots: survival(ngram10) vs model-consistency NLL, vs n-gram
     similarity, vs spelling error. Hard no-repeat shown SEPARATELY.
  2. Kaplan-Meier survival curves for key methods.
  3. Loop-definition robustness: RMST per method across all loop definitions
     (shows the ranking is not tied to one metric — punkt 4).
  4. RMST bar chart with the survival ranking.
  5. Ablation bar charts.
  6. Generalization figure (if a generalize JSON is supplied).

Usage:
  python3 plot_results.py --dir runs/sampling_eval_v6 --out runs/sampling_eval_v6/plots
  python3 plot_results.py --dir runs/sampling_eval_v6 --out runs/.../plots \
      --generalize runs/generalize/distilgpt2.json
"""

import os, sys, json, csv, argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("pip install matplotlib"); sys.exit(1)

COL = {
    "baseline":"#7A7A7A", "sweep_temp":"#9AA0A6", "sweep_topp":"#9AA0A6",
    "sweep_topk":"#9AA0A6", "sweep_typical":"#9AA0A6", "sweep_reppen":"#C77400",
    "sweep_mirostat":"#3B6FA0", "sweep_lookback":"#B07A00",
    "hard_constraint":"#C0392B", "recurrence_risk":"#1A7A4A", "ablation":"#7A4FB0",
}
MK = {
    "baseline":"o","sweep_temp":"o","sweep_topp":"o","sweep_topk":"o",
    "sweep_typical":"o","sweep_reppen":"s","sweep_mirostat":"s",
    "sweep_lookback":"D","hard_constraint":"X","recurrence_risk":"*","ablation":"^",
}
LAB = {
    "adaptive":"RR-adaptive (ours)","risk_only":"RR-risk-only (ours)",
    "rr_fixed_alpha":"RR fixed-alpha","rr_entropy_only":"RR entropy-only",
    "rr_no_top_p":"RR no top-p","rr_narrow_ngram":"RR narrow n-gram",
    "rr_wide_ngram":"RR wide n-gram","rep_penalty_1.3":"Rep.penalty 1.3",
    "rep_penalty_1.5":"Rep.penalty 1.5","mirostat_tau5.0":"Mirostat τ=5",
    "nucleus_p0.95":"Nucleus p=0.95","temp_0.8":"Temp 0.8",
    "typical_p0.9":"Typical 0.9","lookback_a3.0":"Look-back",
    "no_repeat_4gram":"No-repeat 4-gram*","no_repeat_3gram":"No-repeat 3-gram*",
    "greedy":"Greedy",
}

def lab(s): return LAB.get(s, s)

def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                if v in ("", "None"): d[k] = None
                elif v in ("True", "False"): d[k] = (v == "True")
                else:
                    try: d[k] = float(v)
                    except ValueError: d[k] = v
            rows.append(d)
    return rows

def pareto(rows, xk, yk, xl, yl, title, out, ckpt, y_lower_better=True):
    sub = [r for r in rows if r.get("checkpoint") == ckpt
           and isinstance(r.get(xk), float) and isinstance(r.get(yk), float)]
    soft = [r for r in sub if not r.get("hard_constraint")]
    hard = [r for r in sub if r.get("hard_constraint")]

    def frontier(pool):
        F = set()
        for r in pool:
            dom = False
            for q in pool:
                if q is r: continue
                better_x = q[xk] >= r[xk]
                better_y = (q[yk] <= r[yk]) if y_lower_better else (q[yk] >= r[yk])
                strict = (q[xk] > r[xk]) or (
                    (q[yk] < r[yk]) if y_lower_better else (q[yk] > r[yk]))
                if better_x and better_y and strict:
                    dom = True; break
            if not dom: F.add(r["strategy"])
        return F

    Fr = frontier(soft)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    seen = set()
    for r in soft:
        cat = r.get("category", "baseline")
        c = COL.get(cat, "#777"); m = MK.get(cat, "o")
        on = r["strategy"] in Fr
        ax.scatter(r[xk], r[yk], c=c, marker=m, s=150 if on else 45,
                   alpha=1.0 if on else 0.4, edgecolors="white",
                   linewidths=1.5 if on else 0.5, zorder=3)
        if on and r["strategy"] not in seen:
            ax.annotate(lab(r["strategy"]), (r[xk], r[yk]),
                        textcoords="offset points", xytext=(6,4),
                        fontsize=8, color=c, fontweight="bold")
            seen.add(r["strategy"])
    for r in hard:
        ax.scatter(r[xk], r[yk], c="#C0392B", marker="X", s=170, alpha=0.9,
                   edgecolors="white", linewidths=1.5, zorder=4)
        ax.annotate(lab(r["strategy"]), (r[xk], r[yk]),
                    textcoords="offset points", xytext=(6,-12),
                    fontsize=8, color="#C0392B", style="italic")
    ax.set_xlabel(xl, fontsize=11); ax.set_ylabel(yl, fontsize=11)
    ax.set_title(f"{title}  [{ckpt}]", fontsize=12)
    ax.grid(True, alpha=0.2, lw=0.5)
    items = [mpatches.Patch(color=COL[c], label=c.replace("_"," ")) for c in
             ("recurrence_risk","ablation","sweep_reppen","sweep_mirostat",
              "sweep_lookback","baseline","hard_constraint")]
    ax.legend(handles=items, fontsize=7.5, loc="best", framealpha=0.8)
    fig.text(0.01, 0.01, "* Hard no-repeat directly forbids the measured event — shown separately.",
             fontsize=7, color="#777")
    fig.tight_layout(rect=[0,0.04,1,1]); fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  {out}")

def km_plot(km, ckpt, methods, out, max_t=500):
    d = km.get(ckpt, {})
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for s in methods:
        sd = d.get(s)
        if not sd: continue
        cat = ("recurrence_risk" if s in ("adaptive","risk_only") else
               "hard_constraint" if "no_repeat" in s else
               "sweep_reppen" if "rep_pen" in s else
               "sweep_mirostat" if "mirostat" in s else
               "sweep_lookback" if "lookback" in s else "baseline")
        c = COL.get(cat, "#777")
        lw = 2.6 if s in ("adaptive","risk_only") else 1.7
        ls = "--" if "no_repeat" in s else "-"
        t = [0] + sd["km_times"] + [max_t]
        sv = [1.0] + sd["km_survival"] + [sd["km_survival"][-1] if sd["km_survival"] else 1.0]
        ax.step(t, sv, where="post", color=c, lw=lw, ls=ls,
                label=f"{lab(s)} (RMST={sd['rmst']:.0f})")
    ax.set_xlabel("Characters generated", fontsize=11)
    ax.set_ylabel("Fraction loop-free", fontsize=11)
    ax.set_title(f"Loop-onset survival (10-gram)  [{ckpt}]", fontsize=12)
    ax.set_xlim(0, max_t); ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.85)
    ax.grid(True, alpha=0.2, lw=0.5)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  {out}")

def loop_robustness_plot(rob_rows, ckpt, out):
    """RMST per method across all loop definitions — shows ranking robustness."""
    methods = ["greedy","temp_0.8","nucleus_p0.95","rep_penalty_1.3",
               "mirostat_tau5.0","lookback_a3.0","no_repeat_4gram",
               "risk_only","adaptive"]
    defs = ["ngram8","ngram10","ngram12","ngram16","lrs20","compress"]
    sub = {r["strategy"]: r for r in rob_rows if r.get("checkpoint") == ckpt}
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(defs)); w = 0.09
    for i, s in enumerate(methods):
        r = sub.get(s)
        if not r: continue
        vals = [r.get(f"rmst_{d}", 0) for d in defs]
        cat = ("recurrence_risk" if s in ("adaptive","risk_only") else
               "hard_constraint" if "no_repeat" in s else "baseline")
        c = COL.get(cat, "#777")
        hatch = "//" if "no_repeat" in s else None
        ax.bar(x + (i-len(methods)/2)*w, vals, w, label=lab(s), color=c,
               alpha=0.9 if s in ("adaptive","risk_only") else 0.6, hatch=hatch,
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(defs, fontsize=9)
    ax.set_ylabel("RMST (loop-free chars)", fontsize=10)
    ax.set_xlabel("Loop definition", fontsize=10)
    ax.set_title(f"Survival ranking across loop definitions  [{ckpt}]", fontsize=12)
    ax.legend(fontsize=7, ncol=3, loc="lower right", framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.2, lw=0.5)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  {out}")

def rmst_bar(rows, ckpt, out):
    methods = ["greedy","temp_0.8","nucleus_p0.95","typical_p0.9","lookback_a3.0",
               "rep_penalty_1.3","mirostat_tau5.0","no_repeat_4gram",
               "risk_only","adaptive"]
    sub = {r["strategy"]: r for r in rows if r.get("checkpoint") == ckpt}
    data = [(s, sub[s]) for s in methods if s in sub]
    data.sort(key=lambda kv: kv[1].get("rmst_ngram10", 0))
    names = [lab(s) for s,_ in data]
    vals  = [r.get("rmst_ngram10",0) for _,r in data]
    cols  = ["#1A7A4A" if s in ("adaptive","risk_only") else
             "#C0392B" if "no_repeat" in s else
             "#C77400" if "rep_pen" in s else
             "#3B6FA0" if "mirostat" in s else
             "#B07A00" if "lookback" in s else "#7A7A7A" for s,_ in data]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(names))
    ax.barh(y, vals, color=cols, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("RMST — loop-free characters (10-gram, τ=500)", fontsize=10)
    ax.set_title(f"Survival ranking  [{ckpt}]", fontsize=12)
    ax.grid(True, axis="x", alpha=0.2, lw=0.5)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  {out}")

def ablation_bar(rows, ckpt, metric, xl, out, higher=True):
    names_set = ["adaptive","rr_fixed_alpha","risk_only","rr_entropy_only",
                 "rr_no_top_p","rr_narrow_ngram","rr_wide_ngram"]
    sub = {r["strategy"]: r for r in rows if r.get("checkpoint") == ckpt}
    data = [(s, sub[s]) for s in names_set if s in sub]
    data.sort(key=lambda kv: kv[1].get(metric, 0), reverse=higher)
    names = [lab(s) for s,_ in data]
    vals  = [r.get(metric, 0) for _,r in data]
    cols  = ["#1A7A4A" if s in ("adaptive","risk_only") else "#7A4FB0" for s,_ in data]
    fig, ax = plt.subplots(figsize=(8, max(3.2, len(names)*0.55)))
    y = np.arange(len(names))
    ax.barh(y, vals, color=cols, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(xl, fontsize=10)
    ax.set_title(f"Ablation: {xl}  [{ckpt}]", fontsize=11)
    ax.grid(True, axis="x", alpha=0.2, lw=0.5)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  {out}")

def generalize_plot(gpath, out):
    with open(gpath) as f:
        D = json.load(f)
    res = D["results"]
    order = ["greedy","temp_0.9","nucleus_p0.95","rep_penalty_1.3",
             "no_repeat_3gram","risk_only","adaptive"]
    order = [m for m in order if m in res]
    names = [lab(m) for m in order]
    sauc  = [res[m]["survival_auc_tok3"] for m in order]
    nll   = [res[m]["mc_nll_nats"] for m in order]
    cols  = ["#1A7A4A" if m in ("adaptive","risk_only") else
             "#C0392B" if "no_repeat" in m else "#7A7A7A" for m in order]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    y = np.arange(len(names))
    a1.barh(y, sauc, color=cols, alpha=0.85)
    a1.set_yticks(y); a1.set_yticklabels(names, fontsize=9)
    a1.set_xlabel("Survival AUC (3-gram tokens)", fontsize=10)
    a1.set_title(f"{D['model']} — survival", fontsize=11)
    a1.grid(True, axis="x", alpha=0.2, lw=0.5)
    a2.barh(y, nll, color=cols, alpha=0.85)
    a2.set_yticks(y); a2.set_yticklabels(names, fontsize=9)
    a2.set_xlabel("Model-consistency NLL (nats/token)", fontsize=10)
    a2.set_title(f"{D['model']} — consistency", fontsize=11)
    a2.grid(True, axis="x", alpha=0.2, lw=0.5)
    fig.suptitle(f"Generalization to {D['model']} (pretrained subword LM)", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/sampling_eval_v6")
    ap.add_argument("--out", default="runs/sampling_eval_v6/plots")
    ap.add_argument("--n_chars", type=int, default=500)
    ap.add_argument("--generalize", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pareto_rows = load_csv(os.path.join(args.dir, "pareto_data.csv"))
    rob_rows    = load_csv(os.path.join(args.dir, "loop_robustness.csv"))
    with open(os.path.join(args.dir, "survival_curves.json")) as f:
        km = json.load(f)

    ckpts = sorted({r["checkpoint"] for r in pareto_rows})
    key = ["greedy","temp_0.8","nucleus_p0.95","typical_p0.9","lookback_a3.0",
           "rep_penalty_1.3","mirostat_tau5.0","no_repeat_4gram","risk_only","adaptive"]

    for ck in ckpts:
        tag = ck.replace("/","_")
        print(f"\n  {ck}:")
        pareto(pareto_rows, "survival_auc_ngram10", "mc_nll_bpc",
               "Survival AUC (↑ fewer loops)",
               "Model-consistency NLL BPC (↓)",
               "Survival vs model-consistency NLL",
               f"{args.out}/pareto_nll_{tag}.png", ck, y_lower_better=True)
        pareto(pareto_rows, "survival_auc_ngram10", "ngram_sim_4",
               "Survival AUC (↑ fewer loops)",
               "4-gram similarity to held-out text (↑)",
               "Survival vs distributional similarity",
               f"{args.out}/pareto_sim_{tag}.png", ck, y_lower_better=False)
        pareto(pareto_rows, "survival_auc_ngram10", "spelling_error",
               "Survival AUC (↑ fewer loops)",
               "Spelling error rate (↓)",
               "Survival vs spelling error",
               f"{args.out}/pareto_spelling_{tag}.png", ck, y_lower_better=True)
        km_plot(km, ck, key, f"{args.out}/km_{tag}.png", args.n_chars)
        loop_robustness_plot(rob_rows, ck, f"{args.out}/loop_robustness_{tag}.png")
        rmst_bar(pareto_rows, ck, f"{args.out}/rmst_{tag}.png")
        ablation_bar(pareto_rows, ck, "survival_auc_ngram10",
                     "Survival AUC", f"{args.out}/ablation_sauc_{tag}.png", True)
        ablation_bar(pareto_rows, ck, "mc_nll_bpc",
                     "Model-consistency NLL (BPC)", f"{args.out}/ablation_nll_{tag}.png", False)

    if args.generalize and os.path.exists(args.generalize):
        print("\n  generalization:")
        generalize_plot(args.generalize, f"{args.out}/generalize.png")

    print(f"\n  Plots -> {args.out}/")


if __name__ == "__main__":
    main()