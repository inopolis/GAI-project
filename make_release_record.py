#!/usr/bin/env python3
"""
make_release_record.py -- writes release_record.json: SHA-256 of the code, settings, saved results,
generated paper and figures that make up one recorded version, plus the git commit it was built on.

verify_paper_tables.py compares the current files with this record, so any change after the record was
made (code, a result file, a table, the paper source or a figure) is reported.

Usage:  python3 make_release_record.py          # after build_paper.py and make_figures.py
"""
import glob, hashlib, inspect, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

CODE = ["src/decoding.py", "one_step_projection.py", "synthetic_fsm_hazard.py", "budget_ladder.py",
        "verify_bernoulli_bound.py", "frontier_comparison.py", "common_kl_comparison.py", "sampling_eval.py",
        "prompt_conditional_bound.py", "compute_significance.py", "make_figures.py", "build_paper.py",
        "verify_paper_tables.py", "make_release_record.py", "tests/test_budget_ladder.py",
        "tests/test_infeasible_rule.py", "tests/test_fsd_against_official.py", "tests/test_lz_against_reference.py"]
RESULTS = ["budget_ladder.json", "synthetic_fsm_results.json", "functional_form_ablation.json",
           "runs/main_comparison_v7_fsd2_merged/all_results.json", "runs/nonfiction_v4_fsd2_merged/all_results.json",
           "runs/gpt2_full_comparison_v3_merged/all_results.json",
           "runs/modern_llm_comparison_olmo_v3_merged/all_results.json",
           "runs/modern_llm_comparison_mistral_v3_merged/all_results.json",
           "runs/common_kl_comparison_v4_samples/common_kl_comparison.csv",
           "runs/common_kl_comparison_v4_samples/common_kl_samples.csv",
           "runs/common_kl_comparison_v4_samples/prompt_conditional_bound.json"]


def sha(path):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, path), "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    rec = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": {}, "classes": {}}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rec["git_head_when_recorded"] = head
    rec["note"] = ("git_head_when_recorded is the commit the working tree was based on; files listed here may "
                   "have been committed afterwards. The hashes, not the commit, identify the version.")
    rec["working_tree_dirty_when_recorded"] = bool(dirty)
    paths = CODE + RESULTS + ["paper_make.tex", "references.bib"] + \
        sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "paper_src", "*"))) + \
        sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "figures", "*.pdf"))) + \
        sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "runs", "frontier_v1*", "frontier_*.csv")))
    for p in paths:
        if os.path.exists(os.path.join(ROOT, p)):
            rec["files"][p] = sha(p)
    from src import decoding
    for name in ["FSDDecoder", "LZPenaltyDecoder"]:
        rec["classes"][name] = hashlib.sha256(inspect.getsource(getattr(decoding, name)).encode()).hexdigest()
    # the dual-mode branch changed after the language-model runs (rule for an unmet budget): record why
    # the reported runs are unaffected
    rec["language_model_runs_note"] = ("src/decoding.py was changed after the language-model runs only in "
                                       "RecurrenceRiskDecoder._solve_dual_lambda (rule for an unmet budget, "
                                       "failure reporting). verify_paper_tables.py checks that every reported "
                                       "dual run has feasible rate 1.0 and structurally-infeasible rate 0.0, so "
                                       "that branch was not executed.")
    rec["fsd_lz_frozen_before_results"] = "2026-09-27T02:50"
    json.dump(rec, open(os.path.join(ROOT, "release_record.json"), "w"), indent=1)
    print("wrote release_record.json:", len(rec["files"]), "files")


if __name__ == "__main__":
    main()
