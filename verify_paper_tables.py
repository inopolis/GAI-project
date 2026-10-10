#!/usr/bin/env python3
"""
verify_paper_tables.py -- checks that the paper reports exactly what the saved results contain,
and that the saved results were produced by the released code.

  A. paper_make.tex is byte-identical to what build_paper.py generates (no hand-edited tables);
  B. the appendix tables that are kept as LaTeX blocks (full primary comparison, GPT-2,
     non-fiction, OLMo, Mistral, indicator-controller tables) match the saved result files;
  C. headline numbers quoted in the text are recomputed from the saved results;
  D. provenance: every result file is newer than the last change of src/decoding.py;
  E. (--tests) every tests/test_*.py runs as a script and passes (baseline fidelity needs network).

Exit status 0 only if every check passes.   Usage:  python3 verify_paper_tables.py [--tests]
"""
import argparse, csv, glob, json, math, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
FAIL = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  " + detail) if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


def J(path):
    return json.load(open(os.path.join(ROOT, path)))


def table_rows(tex, label):
    i = tex.index("\\label{%s}" % label)
    j = tex.index("\\begin{tabular}", i)
    k = tex.index("\\end{tabular}", j)
    rows = []
    for ln in tex[j:k].split("\n"):
        if "&" in ln and "toprule" not in ln and not ln.lstrip().startswith(("Method", "$\\varepsilon", "Target", "\\textbf{Task")):
            cells = [c.strip().rstrip("\\").strip() for c in ln.rstrip("\\ ").split("&")]
            rows.append(cells)
    return rows


def num(x):
    m = re.search(r"-?\d+\.?\d*", x.replace("\\textbf{", ""))
    return float(m.group()) if m else None


NAME2KEY = {"Greedy": "greedy", "Temperature 0.05": "lt_temp_0.05", "Temperature 0.10": "lt_temp_0.1",
            "Temperature 0.15": "lt_temp_0.15", "Repetition penalty 1.3": "lt_rep_penalty_1.3",
            "SuffixMatch": "lt_suffixmatch_a3.0", "FSD-style": "lt_fsd", "LZ Penalty": "lt_lzpenalty"}


def key_of(label, results):
    label = label.replace("$^{*}$", "").replace("\\textsc{", "").replace("}", "").replace("$", "")
    for k, v in NAME2KEY.items():
        if label.startswith(k):
            return v
    if "No-repeat" in label:
        c = [x for x in results if "no_repeat" in x]
        return c[0] if c else None
    if "rr-fixed" in label:
        return "lt_risk_only"
    if "rr-adaptive" in label:
        return "lt_adaptive"
    if "rr-dual" in label:
        return "lt_dual_eps0.05" if "0.05" in label else "lt_dual_eps0.01"
    return None


def check_results_table(tex, label, path, cols, group=None):
    d = J(path)
    if isinstance(d, dict):
        d = d[group] if group in d else d[list(d)[0]]
    R = {r["strategy"]: r for r in d}
    bad = []
    for c in table_rows(tex, label):
        k = key_of(c[0], R)
        if k is None or k not in R:
            continue
        r = R[k]
        for idx, (field, scale, tol) in cols.items():
            if idx >= len(c) or c[idx] in ("---", ""):
                continue
            if field == "loops":
                kk, nn = c[idx].split("/")
                ok = int(nn) == r["n_samples"] and abs(int(kk) - r["loop_rate_persistent"] * r["n_samples"]) < 0.5
            else:
                f = field if field in r else field.replace("bits", "bpc")
                if f not in r:
                    continue
                ok = abs(num(c[idx]) - scale * r[f]) < tol
            if not ok:
                bad.append((c[0], idx, c[idx]))
    check(f"table {label} matches {path}", not bad, str(bad[:3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=os.path.join(ROOT, "paper_make.tex"))
    ap.add_argument("--tests", action="store_true")
    a = ap.parse_args()
    tex = open(a.paper).read()

    # A. generated == committed
    import build_paper
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False) as f:
        tmp = f.name
    gen = build_paper.build(tmp)
    check("A. paper_make.tex equals the output of build_paper.py", gen == tex)

    # B. appendix tables
    check_results_table(tex, "tab:main", "runs/main_comparison_v7_fsd2_merged/all_results.json",
                        {1: ("loops", 1, 0), 2: ("rmst_persistent", 1, 0.6), 3: ("mc_nll_bpc_mean", 1, 0.0006)}, "cosine")
    check_results_table(tex, "tab:gpt2full", "runs/gpt2_full_comparison_v3_merged/all_results.json",
                        {2: ("loop_rate_persistent", 100, 0.06), 3: ("rmst_persistent", 1, 0.06), 4: ("mc_nll_bits_mean", 1, 0.0006)}, "gpt2")
    check_results_table(tex, "tab:generalization", "runs/nonfiction_v4_fsd2_merged/all_results.json",
                        {2: ("loop_rate_persistent", 100, 0.06), 3: ("rmst_persistent", 1, 0.6), 4: ("mc_nll_bpc_mean", 1, 0.0006),
                         5: ("kl_bits_mean", 1, 0.0006)})
    for lab, path in [("tab:olmofull", "runs/modern_llm_comparison_olmo_v3_merged/all_results.json"),
                      ("tab:mistralfull", "runs/modern_llm_comparison_mistral_v3_merged/all_results.json")]:
        check_results_table(tex, lab, path, {1: ("loop_rate_persistent", 100, 0.06), 2: ("rmst_persistent", 1, 0.06),
                                             3: ("mc_nll_bits_mean", 1, 0.0006), 4: ("kl_bits_mean", 1, 0.0006)})
    # indicator-controller tables
    syn = J("synthetic_fsm_results.json")
    rows = [c for c in table_rows(tex, "tab:oracle") if re.match(r"^0\.\d+$", c[0])]
    bad = 0
    for r, c in zip(syn["rows"], rows):
        exp = [r["eps"], r["local_kl"], r["hazard_kl"], r["oracle_kl_per_step"]]
        got = [num(c[i]) for i in range(4)]
        bad += sum(abs(x - y) > 6e-4 for x, y in zip(exp, got))
    check("table tab:oracle matches synthetic_fsm_results.json", bad == 0 and len(rows) == len(syn["rows"]))

    # C. headline numbers recomputed
    d = J("budget_ladder.json")
    lam = [r["rules"]["lam"]["ratio"] for sy in d["systems"] for r in sy["grid"]]
    ab = [r["rules"]["abs"]["ratio"] for sy in d["systems"] for r in sy["grid"] if r["rules"]["abs"].get("reachable", True)]
    check("C. fixed price within 12.5% of minimum in all systems", max(lam) < 1.126 and "12.5\\%" in tex, f"max={max(lam):.4f}")
    check("C. fixed budget never below 1.3x on the grids", min(ab) > 1.3, f"min={min(ab):.3f}")
    check("C. fixed budget > 200x at the smallest reduction", d["small_distortion"][-1]["abs"]["exact_ratio"] > 200)
    base = d["systems"][0]
    r60 = [r for r in base["grid"] if abs(r["eps"] - 0.60) < 1e-9][0]["rules"]
    check("C. text: 12.4 -> 4.8 times at eps=0.6", abs(r60["abs"]["ratio"] - 12.41) < 0.01 and abs(r60["rel"]["ratio"] - 4.81) < 0.01)
    r3 = [s for s in d["systems"] if s["name"].startswith("persistent R=3")][0]
    z = [r["rules"]["abs"]["infeasible_mass"] for r in r3["grid"]]
    ab3 = [r["rules"]["abs"]["ratio"] for r in r3["grid"]]
    check("C. R=3 system: no infeasible node, FB 2.7-7.1x", max(z) == 0 and abs(min(ab3) - 2.74) < 0.01 and abs(max(ab3) - 7.11) < 0.01)
    check("C. Spearman 0.82 over 244 runs", d["spearman_logratio_vs_log1pCV2"]["n"] == 244 and abs(d["spearman_logratio_vs_log1pCV2"]["rho"] - 0.82) < 0.005)
    pcb = J("runs/common_kl_comparison_v4_samples/prompt_conditional_bound.json")
    ex = [100 * (1 - pcb[k]["bound_prompt_conditional"] / pcb[k]["path_kl_bits"]) for k in pcb if k != "_reference"]
    check("C. excess share >= 98% for every decoder", min(ex) >= 98.0, f"min={min(ex):.2f}")
    p = pcb["_reference"].get("loop_rate", None) if isinstance(pcb["_reference"], dict) else None
    rows = {r["strategy"]: r for r in csv.DictReader(open(os.path.join(ROOT, "runs/common_kl_comparison_v4_samples/common_kl_comparison.csv")))}
    p = float(rows["temperature_0.10"]["loop_rate"])
    check("C. rejection cost 0.49 bits / 1.4 generations at p=0.287", abs(-math.log2(1 - p) - 0.488) < 0.005 and abs(1 / (1 - p) - 1.40) < 0.01)
    from scipy.stats import beta
    check("C. upper limit 2.4% for 0/150", abs(1 - 0.025 ** (1 / 150) - 0.0243) < 5e-4)

    fr = {}
    for f in glob.glob(os.path.join(ROOT, "runs/frontier_v1*/frontier_rows.csv")):
        for r in csv.DictReader(open(f)):
            fr[r["strategy"]] = r
    if fr:
        kl = lambda k: float(fr[k]["kl_bits_total_mean"])
        du = [fr[k] for k in fr if k.startswith("rr_dual")]
        check("C. frontier: 0/75 at lam=3 (38.6 bits), 0/75 at eps=0.5 (1696 bits), 44x and 92x",
              fr["rr_fixed_lam3"]["loop_count"] == "0" and fr["rr_dual_eps0.5"]["loop_count"] == "0"
              and abs(kl("rr_fixed_lam3") - 38.6) < 0.1 and abs(kl("rr_dual_eps0.5") - 1696.0) < 0.5
              and round(kl("rr_dual_eps0.5") / kl("rr_fixed_lam3")) == 44 and round(kl("rr_dual_eps0.5") / kl("rr_fixed_lam2")) == 92)
        check("C. frontier: eps=0.01 costs 23x lam=12; reference 29/75; lam=2 2/75 at 18.5 bits",
              round(kl("rr_dual_eps0.01") / kl("rr_fixed_lam12")) == 23 and fr["temperature_0.10"]["loop_count"] == "29"
              and fr["rr_fixed_lam2"]["loop_count"] == "2" and abs(kl("rr_fixed_lam2") - 18.5) < 0.1)
        check("C. frontier: inactive at 60-80% of steps, p90 29-163, max 115-437",
              all(0.595 < float(r["lam_frac_zero"]) < 0.805 for r in du)
              and round(min(float(r["lam_p90"]) for r in du)) == 29 and round(max(float(r["lam_p90"]) for r in du)) == 163
              and round(min(float(r["lam_max"]) for r in du)) == 115 and round(max(float(r["lam_max"]) for r in du)) == 437)
        check("C. frontier: eps=0.65 0/75 at 1356 bits = 35x lam=3 = 73x lam=2; lam=1 4/75 at 5.0 bits",
              fr["rr_dual_eps0.65"]["loop_count"] == "0" and abs(kl("rr_dual_eps0.65") - 1355.7) < 0.2
              and round(kl("rr_dual_eps0.65") / kl("rr_fixed_lam3")) == 35 and round(kl("rr_dual_eps0.65") / kl("rr_fixed_lam2")) == 73
              and fr["rr_fixed_lam1"]["loop_count"] == "4" and abs(kl("rr_fixed_lam1") - 5.0) < 0.05)
        check("C. upper limit 4.8% for 0/75", abs(1 - 0.025 ** (1 / 75) - 0.048) < 5e-4)

    # D. provenance
    t_code = os.path.getmtime(os.path.join(ROOT, "src/decoding.py"))
    stale = []
    for pat in ["runs/main_comparison_v7_fsd2_merged/all_results.json", "runs/nonfiction_v4_fsd2_merged/all_results.json",
                "runs/gpt2_full_comparison_v3_merged/all_results.json", "runs/modern_llm_comparison_olmo_v3_merged/all_results.json",
                "runs/modern_llm_comparison_mistral_v3_merged/all_results.json",
                "runs/common_kl_comparison_v4_samples/common_kl_comparison.csv", "budget_ladder.json"] + \
               [os.path.relpath(x, ROOT) for x in glob.glob(os.path.join(ROOT, "runs/frontier_v1*/frontier_rows.csv"))]:
        fp = os.path.join(ROOT, pat)
        if os.path.exists(fp) and os.path.getmtime(fp) < t_code:
            stale.append(pat)
    check("D. every result file is newer than the last change of src/decoding.py", not stale, str(stale))
    st = subprocess.run(["git", "status", "--porcelain", "--", "src", "tests", "*.py"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    check("D. source files committed (git working tree clean for src/, tests/, *.py)", st == "", "uncommitted: " + st.replace("\n", "; ")[:200])

    # E. tests
    if a.tests:
        for t in sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))):
            r = subprocess.run([sys.executable, t], cwd=ROOT, capture_output=True, text=True)
            check("E. " + os.path.basename(t), r.returncode == 0, (r.stdout + r.stderr)[-300:])

    print("\n%d check(s) failed" % len(FAIL) if FAIL else "\nall checks passed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
