"""
Post-hoc sensitivity and Pareto analysis, built entirely from artifacts
already written by prior runs -- no regeneration, no model loading.

Addresses two of the professor's standing requirements ("sensitivity to
detector thresholds and generation length, and Pareto plots of loop risk
versus KL/quality rather than only single operating points") for the parts
of the project where the needed raw material (per-sample onset arrays,
and/or raw generated text) was already saved:

  1. Generation-length sensitivity: for every experiment that saved
     per-sample onset positions (onsets_persistent), recompute loop_rate at
     several truncated horizons instead of only the final one. This is
     exact -- onset position already tells us the earliest point of loop
     entry, so truncating the horizon and re-censoring is not an
     approximation.

  2. Detector-threshold (R, P_max) sensitivity: for the primary char-level
     model (main_comparison_v3_final), which saved raw generated TEXT
     (samples_cosine.txt), recompute persistent_loop_onset() at a grid of
     (R, P_max) around the frozen (5, 60) directly from that text -- exact,
     because the char-level event operates on the literal character stream
     the model produced, so there is no retokenization-fidelity question.

     NOT attempted here for the Mistral/OLMo (Block 2) subword runs: only
     DECODED text was saved for those, and re-tokenizing decoded text to
     recover token ids is not reliably faithful for a subword tokenizer --
     empirically checked before writing this script (see chat record):
     18% of Mistral samples and 2% of OLMo samples produced a different
     token count than the original generation when the saved text was
     re-tokenized from scratch. Silently absorbing that error into a
     sensitivity table that feeds the paper would be worse than not having
     the table; genuine (R, period_max) sensitivity for Block 2 needs a
     rerun with raw token ids saved, which is a separate, explicit,
     compute-cost decision, not something to fold in here.

  3. Pareto plots (KL bits vs. loop rate) for every experiment that has
     BOTH a per-config KL number and a per-config loop rate in the same
     table: the common-distortion GPT-2 comparison (tab:commonkl) and the
     Block 2 Mistral/OLMo full comparisons. main_comparison_v3_final is
     NOT included here since it does not carry a KL column (it predates
     the common-distortion instrumentation) -- plot_results.py's existing
     pareto() already covers its own axes (NLL/spelling/n-gram similarity
     vs. survival AUC).
"""
import os
import re
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sampling_eval import persistent_loop_onset, cluster_bootstrap_ci
from modern_llm_comparison import persistent_loop_onset_tokens

OUT_DIR = "runs/sensitivity_pareto"
os.makedirs(OUT_DIR, exist_ok=True)

HEADER_RE = re.compile(
    r"^\[(?P<strategy>[^\]]+)\]\[(?P<model>[^\]]+)\] prompt='(?P<prompt>.*?)' seed=(?P<seed>\d+)$",
    re.M,
)
SEP = "-" * 60 + "\n"


def parse_samples_file(path):
    """Returns list of (strategy, prompt, seed, text) in file order."""
    content = open(path, encoding="utf-8").read()
    matches = list(HEADER_RE.finditer(content))
    samples = []
    for i, m in enumerate(matches):
        body_start = content.index(SEP, m.end()) + len(SEP)
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[body_start:body_end]
        if text.endswith("\n\n"):
            text = text[:-2]
        elif text.endswith("\n"):
            text = text[:-1]
        samples.append((m.group("strategy"), m.group("prompt"), int(m.group("seed")), text))
    return samples


# ---------------------------------------------------------------------------
# 1. Generation-length sensitivity (exact, from already-saved onset arrays)
# ---------------------------------------------------------------------------

def genlen_sensitivity_from_all_results(path, key, horizons, label):
    d = json.load(open(path))
    rows = d[key]
    out = []
    for r in rows:
        onsets = r.get("onsets_persistent")
        groups = r.get("groups")
        if not onsets:
            continue
        if not groups:
            groups = list(range(len(onsets)))
        onsets = np.asarray(onsets, dtype=float)
        groups = np.asarray(groups)
        for h in horizons:
            fired = ((onsets >= 0) & (onsets <= h)).astype(float)
            mean, lo, hi = cluster_bootstrap_ci(fired, groups)
            out.append({
                "experiment": label, "strategy": r["strategy"], "horizon": h,
                "loop_rate": mean, "ci_lo": lo, "ci_hi": hi,
                "n": len(onsets),
            })
    return out


def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# 2. Detector-threshold (R, P_max) sensitivity -- char-level model only
# ---------------------------------------------------------------------------

def threshold_sensitivity_gpt2char(samples_path, R_grid, Pmax_grid):
    samples = parse_samples_file(samples_path)
    by_strategy = {}
    for strategy, prompt, seed, text in samples:
        by_strategy.setdefault(strategy, []).append((prompt, text))

    out = []
    for strategy, items in by_strategy.items():
        prompts = sorted({p for p, _ in items})
        prompt_to_group = {p: i for i, p in enumerate(prompts)}
        groups = np.array([prompt_to_group[p] for p, _ in items])
        for R in R_grid:
            for Pmax in Pmax_grid:
                if Pmax < 2:
                    continue
                fired = np.array([
                    1.0 if persistent_loop_onset(text, P_max=Pmax, R=R) >= 0 else 0.0
                    for _, text in items
                ])
                mean, lo, hi = cluster_bootstrap_ci(fired, groups)
                out.append({
                    "strategy": strategy, "R": R, "P_max": Pmax,
                    "loop_rate": mean, "ci_lo": lo, "ci_hi": hi,
                    "n": len(items),
                })
    return out


def threshold_sensitivity_subword(all_results_path, key, R_grid, Pmax_grid, label):
    """Exact (R, period_max) sensitivity for a Block-2 subword-model run --
    uses the raw generated token ids saved in all_results.json (added
    specifically for this), not a retokenization of decoded text."""
    rows = json.load(open(all_results_path))[key]
    out = []
    for r in rows:
        gen_ids_all = r.get("gen_ids")
        if not gen_ids_all:
            continue
        groups = np.asarray(r.get("groups") or list(range(len(gen_ids_all))))
        for R in R_grid:
            for Pmax in Pmax_grid:
                fired = np.array([
                    1.0 if persistent_loop_onset_tokens(ids, R, Pmax) >= 0 else 0.0
                    for ids in gen_ids_all
                ])
                mean, lo, hi = cluster_bootstrap_ci(fired, groups)
                out.append({
                    "experiment": label, "strategy": r["strategy"], "R": R, "P_max": Pmax,
                    "loop_rate": mean, "ci_lo": lo, "ci_hi": hi,
                    "n": len(gen_ids_all),
                })
    return out


# ---------------------------------------------------------------------------
# 3. Pareto plots: KL bits vs. loop rate
# ---------------------------------------------------------------------------

def pareto_plot(points, title, out_path, x_label="KL vs. shared reference (bits)",
                 y_label="Loop rate"):
    """points: list of (name, kl_bits, loop_rate). Only Pareto-frontier points
    get an inline text label (matching plot_results.py's existing pareto()
    convention) -- dense clusters of dominated points get a small numbered
    marker instead, keyed to a legend, since inline names on every point
    become unreadable overlapping text as soon as more than 2-3 points land
    near the same (KL, loop_rate)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = np.array([p[1] for p in points])
    ys = np.array([p[2] for p in points])

    # Pareto frontier: a point is dominated if another point has BOTH
    # lower-or-equal KL and lower-or-equal loop rate, with at least one
    # strictly lower.
    is_dominated = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:
                continue
            if xs[j] <= xs[i] and ys[j] <= ys[i] and (xs[j] < xs[i] or ys[j] < ys[i]):
                is_dominated[i] = True
                break

    # Frontier points can still coincide (e.g. two configs that are the same
    # sampling distribution by construction land on the exact same (KL,
    # loop_rate)) -- stagger their label offsets so the text doesn't overlap.
    seen_coords = {}
    dominated_legend = []
    dom_idx = 0
    for i, (name, x, y) in enumerate(points):
        if not is_dominated[i]:
            key = (round(x, 3), round(y, 3))
            k = seen_coords.get(key, 0)
            seen_coords[key] = k + 1
            xytext = (6, 5 + 13 * k)
            ax.scatter(x, y, c="#1f77b4", marker="o", s=70, zorder=3)
            ax.annotate(name, (x, y), textcoords="offset points", xytext=xytext,
                        fontsize=8, fontweight="bold", color="#1f77b4")
        else:
            dom_idx += 1
            ax.scatter(x, y, c="#999999", marker="x", s=50, zorder=2)
            ax.annotate(str(dom_idx), (x, y), textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color="#666666")
            dominated_legend.append(f"{dom_idx}: {name}")

    frontier = sorted([(x, y) for i, (n, x, y) in enumerate(points) if not is_dominated[i]])
    if frontier:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, "--", color="#1f77b4", alpha=0.5, zorder=1,
                 label="Pareto frontier (not dominated)")

    if dominated_legend:
        ax.text(1.02, 1.0, "Dominated points:\n" + "\n".join(dominated_legend),
                 transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
                 color="#555555")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return [points[i][0] for i in range(len(points)) if not is_dominated[i]]


def pareto_from_common_kl(path):
    rows = json.load(open(path))
    return [(r["strategy"], r["kl_bits_mean"], r["loop_rate"]) for r in rows]


def pareto_from_modern_llm(path, key):
    rows = json.load(open(path))[key]
    return [(r["strategy"], r["kl_bits_mean"], r["loop_rate_persistent"]) for r in rows]


def main():
    print("=== 1. Generation-length sensitivity ===")
    all_genlen = []
    all_genlen += genlen_sensitivity_from_all_results(
        "runs/main_comparison_v3_final/all_results.json", "cosine",
        horizons=[100, 200, 300, 400, 500], label="gpt2char_main")
    all_genlen += genlen_sensitivity_from_all_results(
        "runs/modern_llm_comparison_mistral/all_results.json", "mistralai/Mistral-7B-v0.3",
        horizons=[30, 60, 90, 120, 150], label="mistral")
    all_genlen += genlen_sensitivity_from_all_results(
        "runs/modern_llm_comparison_olmo/all_results.json", "allenai/OLMo-2-0425-1B",
        horizons=[30, 60, 90, 120, 150], label="olmo")
    write_csv(all_genlen, os.path.join(OUT_DIR, "genlen_sensitivity.csv"),
              ["experiment", "strategy", "horizon", "loop_rate", "ci_lo", "ci_hi", "n"])

    print("\n=== 2. Detector-threshold (R, P_max) sensitivity: GPT-2 char model ===")
    thresh = threshold_sensitivity_gpt2char(
        "runs/main_comparison_v3_final/samples_cosine.txt",
        R_grid=[3, 4, 5, 6, 7], Pmax_grid=[30, 45, 60, 90, 120])
    write_csv(thresh, os.path.join(OUT_DIR, "detector_threshold_sensitivity_gpt2char.csv"),
              ["strategy", "R", "P_max", "loop_rate", "ci_lo", "ci_hi", "n"])

    print("\n=== 3. Pareto plots (KL vs. loop rate) ===")
    gpt2_kl_points = pareto_from_common_kl("runs/common_kl_comparison/common_kl_comparison.json")
    frontier1 = pareto_plot(gpt2_kl_points, "GPT-2 char model: common-distortion comparison",
                             os.path.join(OUT_DIR, "pareto_gpt2char_commonkl.png"))
    print("  frontier:", frontier1)

    mistral_points = pareto_from_modern_llm(
        "runs/modern_llm_comparison_mistral/all_results.json", "mistralai/Mistral-7B-v0.3")
    frontier2 = pareto_plot(mistral_points, "Mistral-7B-v0.3: full decoder comparison",
                             os.path.join(OUT_DIR, "pareto_mistral.png"))
    print("  frontier:", frontier2)

    olmo_points = pareto_from_modern_llm(
        "runs/modern_llm_comparison_olmo/all_results.json", "allenai/OLMo-2-0425-1B")
    frontier3 = pareto_plot(olmo_points, "OLMo-2-0425-1B: full decoder comparison",
                             os.path.join(OUT_DIR, "pareto_olmo.png"))
    print("  frontier:", frontier3)

    print(f"\nAll outputs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
