"""
make_figures.py -- figures of the paper, generated from saved result files only.

  fig_ladder.pdf     small system: KL spent relative to the event-level optimum, by rule
  fig_waste_law.pdf  small system: second-order waste law (convergence and scatter)
  fig_frontier.pdf   primary model: loop rate vs KL for fixed-multiplier and fixed-budget rules
                     (only if runs/frontier_v1/frontier_rows.csv exists)

Usage: python3 make_figures.py [--ladder budget_ladder.json] [--frontier runs/frontier_v1] [--out figures]
"""
import argparse, csv, json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colour-blind-safe palette
C = {"abs": "#D55E00", "rel": "#E69F00", "sched": "#CC79A7", "lam": "#0072B2", "affine": "#000000"}
LAB = {"abs": "fixed budget (re-solved each step)", "rel": "fixed budget, always feasible",
       "sched": "best history-independent budget schedule", "lam": "fixed multiplier (exponential)",
       "affine": "optimum (affine, one multiplier)"}
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "figure.dpi": 150, "savefig.bbox": "tight"})


def fig_ladder(d, out):
    base = d["systems"][0]
    p = base["p"]
    rows = list(base["grid"])
    eps = [r["eps"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.5, 1]})
    for k in ["abs", "rel", "sched", "lam", "affine"]:
        ys = [r["rules"].get(k, {}).get("ratio") for r in rows]
        xs = [e for e, y in zip(eps, ys) if y is not None and y == y]
        ys = [y for y in ys if y is not None and y == y]
        ax[0].plot(xs, ys, marker="o", ms=3, lw=1.4, color=C[k], label=LAB[k],
                   ls="--" if k == "affine" else "-")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].invert_xaxis()
    ax[0].set_xlabel(r"target loop probability $\epsilon$ (reference $p=%.2f$)" % p)
    ax[0].set_ylabel("KL spent / minimum KL  $d_{\\mathrm{Ber}}(\\epsilon\\Vert p)$")
    ax[0].set_title("(a) cost relative to the optimum")
    ax[0].axhline(1, color="0.7", lw=0.6)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="lower left", ncol=2, bbox_to_anchor=(0.02, -0.02))
    # panel b: share of the fixed-budget controller's KL at infeasible nodes
    inf = [100 * r["rules"]["abs"]["kl_share_infeasible"] for r in rows]
    mass = [100 * r["rules"]["abs"]["infeasible_mass"] for r in rows]
    ax[1].plot(eps, inf, marker="o", ms=3, color=C["abs"], label="share of KL spent at infeasible nodes")
    ax[1].plot(eps, mass, marker="s", ms=3, color="0.4", ls=":", label="share of node visits that are infeasible")
    ax[1].set_xscale("log"); ax[1].invert_xaxis()
    ax[1].set_xlabel(r"target loop probability $\epsilon$")
    ax[1].set_ylabel("%")
    ax[1].set_title("(b) fixed budget: infeasibility")
    ax[1].legend(frameon=False, fontsize=6.8, loc="upper left")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(os.path.join(out, "fig_ladder.pdf"))
    plt.close(fig)


def fig_waste(d, out):
    conv = d["small_distortion"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6))
    D = [c["D"] for c in conv]
    for rule, col, name in [("abs", C["abs"], "fixed budget"), ("rel", C["rel"], "always-feasible budget"),
                            ("lam", C["lam"], "fixed multiplier")]:
        ax[0].plot(D, [c[rule]["exact_ratio"] for c in conv], marker="o", ms=3, color=col, lw=1.4,
                   label=f"{name}: exact")
        ax[0].plot(D, [c[rule]["R_pred"] for c in conv], color=col, lw=0.9, ls=":",
                   label=f"{name}: $1+\\mathrm{{CV}}^2$" if rule != "lam" else None)
    lim = d["fixed_budget_limit"]["one_over_phi_max"]
    ax[0].axhline(lim, color=C["abs"], lw=0.8, ls="--")
    ax[0].text(2e-3, lim * 1.1, r"$1/\varphi_{\max}=%.0f$" % lim, fontsize=6.8, color=C["abs"])
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"total reduction $D = p-\epsilon$")
    ax[0].set_ylabel("KL / minimum KL")
    ax[0].set_title("(a) small reductions: ratio saturates at $1/\\varphi_{\\max}$")
    h0, l0 = ax[0].get_legend_handles_labels()
    fig.legend(h0, l0, frameon=False, fontsize=6.8, loc="lower left", ncol=2, bbox_to_anchor=(0.02, -0.02))
    xs, ys, cs = [], [], []
    for sy in d["systems"]:
        for r in sy["grid"]:
            for k in ["abs", "rel", "lam"]:
                e = r["rules"].get(k)
                if e and e.get("reachable", True) and "ratio" in e:
                    xs.append(1 + e["cv2_lambda"]); ys.append(e["ratio"]); cs.append(C[k])
    for k, nm in [("abs", "fixed budget"), ("rel", "always-feasible budget"), ("lam", "fixed multiplier")]:
        sel = [i for i, c in enumerate(cs) if c == C[k]]
        ax[1].scatter([xs[i] for i in sel], [ys[i] for i in sel], s=7, c=C[k], alpha=0.7, lw=0, label=nm)
    ax[1].legend(frameon=False, fontsize=6.5, loc="upper left")
    lim = [1, max(xs) * 1.2]
    ax[1].plot(lim, lim, color="0.5", lw=0.8, ls="--")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    rho = d["spearman_logratio_vs_log1pCV2"]
    ax[1].set_xlabel(r"dispersion of implied multipliers, $1+\mathrm{CV}^2$")
    ax[1].set_ylabel("KL / minimum KL (exact)")
    ax[1].set_title(f"(b) all {rho['n']} runs, 8 systems; Spearman {rho['rho']:.2f}")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(os.path.join(out, "fig_waste_law.pdf"))
    plt.close(fig)


def fig_frontier(dirpath, out):
    import glob, re
    paths = sorted(glob.glob(dirpath + "*/frontier_rows.csv")) + sorted(glob.glob(os.path.join(dirpath, "frontier_rows.csv")))
    if not paths:
        print("no frontier results yet under", dirpath)
        return
    rows = {}
    for pth in dict.fromkeys(paths):
        for r in csv.DictReader(open(pth)):
            rows[r["strategy"]] = r
    num = lambda s: float(re.search(r"(?:lam|eps)([0-9.]+)", s).group(1))
    fixed = sorted([r for k, r in rows.items() if k.startswith("rr_fixed")], key=lambda r: num(r["strategy"]))
    dual = sorted([r for k, r in rows.items() if k.startswith("rr_dual") and num(k) < 1], key=lambda r: -num(r["strategy"]))
    ref = rows.get("temperature_0.10")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for fam, col, mk, name in [(fixed, C["lam"], "o", "fixed price $\\lambda$"),
                                (dual, C["abs"], "s", "fixed budget $\\varepsilon<1$ (dual)")]:
        ax[0].plot([float(r["kl_bits_total_mean"]) for r in fam], [100 * float(r["loop_rate"]) for r in fam],
                   marker=mk, ms=4, color=col, lw=1.2, label=name)
    if ref:
        y0 = 100 * float(ref["loop_rate"])
        ax[0].axhline(y0, color="0.6", lw=0.7, ls=":")
        ax[0].text(0.98, y0 - 2.0, "reference = budget $\\varepsilon=1$ (KL $=0$)", fontsize=6.5, color="0.4", ha="right",
                   transform=ax[0].get_yaxis_transform())
    ax[0].set_xscale("log")
    ax[0].set_xlabel("path KL per 500-character generation (bits)")
    ax[0].set_ylabel("generations with a loop (%)")
    ax[0].set_title("(a) loop rate against distortion")
    ax[0].legend(frameon=False, fontsize=7, loc="center left")
    xs = list(range(len(dual)))
    ax[1].plot(xs, [float(r["lam_mean"]) for r in dual], "o", color=C["abs"], label="mean")
    ax[1].plot(xs, [float(r["lam_p90"]) for r in dual], "s", color=C["abs"], mfc="none", label="90th percentile")
    ax[1].plot(xs, [float(r["lam_max"]) for r in dual], "^", color=C["abs"], mfc="none", label="maximum")
    for x, r in zip(xs, dual):
        if x % 2 == 0:
            ax[1].text(x, 1.25, "%.0f%%" % (100 * float(r["lam_frac_zero"])), ha="center", fontsize=6, color="0.35")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1.0, 1500)
    ax[1].set_xticks(xs[::2])
    ax[1].set_xticklabels([("%g" % num(r["strategy"])) for r in dual][::2], fontsize=7)
    ax[1].set_xlabel(r"per-step budget $\varepsilon$   (grey: share of steps with $\lambda_t=0$)")
    ax[1].set_ylabel(r"per-step multiplier $\lambda_t$")
    ax[1].set_title("(b) multipliers realised by the budget rule")
    ax[1].legend(frameon=False, fontsize=6.5, loc="upper left", ncol=3, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_frontier.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="budget_ladder.json")
    ap.add_argument("--frontier", default="runs/frontier_v1")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = json.load(open(a.ladder))
    fig_ladder(d, a.out)
    fig_waste(d, a.out)
    fig_frontier(a.frontier, a.out)
    print("figures written to", a.out)
