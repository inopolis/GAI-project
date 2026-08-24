"""
merge_sampling_eval_runs.py

sampling_eval.py's --only mode overwrites metrics_*.csv, pareto_data.csv,
loop_robustness.csv, survival_curves.json, all_results.json, and
samples_*.txt with ONLY the subset of configs it was just asked to run --
it does not read and merge whatever was already in --out_dir. Pointing
--only at the SAME --out_dir as an earlier full run therefore silently
destroys every row/sample the earlier run had for the configs NOT in this
--only subset. (The tool's own printed message after an --only run --
"Merge these rows into the full run's CSVs, then recompute comparisons" --
says as much; this script is that missing merge step.)

This combines two or more sampling_eval.py output directories for the SAME
checkpoint into one complete set of files, recomputing every derived
artifact (pareto_data.csv, loop_robustness.csv, survival_curves.json,
method_vs_method.csv, metrics_<name>.csv) from the merged rows via
sampling_eval.py's own functions, so the merged output is byte-for-byte
what a single full run would have produced -- not a hand-patched CSV.

If a strategy name appears in more than one input directory, the LATER
directory (later in --runs) wins and a warning is printed; nothing is
silently dropped.

Usage
-----
  python3 merge_sampling_eval_runs.py \
      --runs runs/main_comparison_v3_missing9 runs/main_comparison_v3 \
      --out_dir runs/main_comparison_v3_merged --n_chars 500

Writes the complete merged set to --out_dir, WITHOUT touching either input
directory, so you can inspect it before replacing anything.
"""
import argparse, json, os, re, sys, csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import ensure_dir
from sampling_eval import (write_pareto, write_loop_robustness,
                           write_survival_curves, write_method_vs_method,
                           SCALARS)

HDR_RE = re.compile(r"^\[([^\]]+)\]\[([^\]]+)\] prompt='(.*?)' seed=(\d+)\n-+\n(.*)$", re.S)


def load_all_results(run_dir):
    path = os.path.join(run_dir, "all_results.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- is {run_dir} a sampling_eval.py output dir?")
    return json.load(open(path))


def parse_samples(run_dir, checkpoint_name):
    """Returns {strategy: [raw block text, ...]} for one checkpoint's samples file."""
    path = os.path.join(run_dir, f"samples_{checkpoint_name}.txt")
    out = {}
    if not os.path.exists(path):
        return out
    text = open(path, encoding="utf-8").read()
    blocks = re.split(r"\n\n(?=\[)", text)
    for b in blocks:
        b = b.strip("\n")
        m = HDR_RE.match(b)
        if not m:
            continue
        strat = m.group(1)
        out.setdefault(strat, []).append(b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="sampling_eval.py output directories to merge, in "
                         "priority order (later wins on strategy-name conflicts).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_chars", type=int, default=500,
                    help="Must match the --n_chars the original run(s) used "
                         "(needed to recompute survival curves/RMST correctly).")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    merged_rows = {}     # checkpoint -> {strategy -> row}
    merged_source = {}   # checkpoint -> {strategy -> source run_dir}
    all_checkpoints = set()

    for run_dir in args.runs:
        data = load_all_results(run_dir)
        for ckpt, rows in data.items():
            all_checkpoints.add(ckpt)
            merged_rows.setdefault(ckpt, {})
            merged_source.setdefault(ckpt, {})
            for row in rows:
                strat = row["strategy"]
                if strat in merged_rows[ckpt]:
                    print(f"  NOTE: '{strat}' ({ckpt}) present in both "
                          f"{merged_source[ckpt][strat]} and {run_dir} -- "
                          f"using {run_dir} (later wins).")
                merged_rows[ckpt][strat] = row
                merged_source[ckpt][strat] = run_dir

    all_results = {ckpt: list(strat_map.values()) for ckpt, strat_map in merged_rows.items()}

    for ckpt in all_checkpoints:
        n = len(all_results[ckpt])
        print(f"  {ckpt}: {n} strategies merged "
              f"({', '.join(r['strategy'] for r in all_results[ckpt])})")

    # ---- merged samples_<ckpt>.txt: pull each winning strategy's blocks
    # from the run_dir that actually won it, so no stale/overridden text
    # from a losing run ends up in the merged file. ----
    for ckpt in all_checkpoints:
        parsed_by_dir = {}
        out_path = os.path.join(args.out_dir, f"samples_{ckpt}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for strat, row in merged_rows[ckpt].items():
                src = merged_source[ckpt][strat]
                if src not in parsed_by_dir:
                    parsed_by_dir[src] = parse_samples(src, ckpt)
                blocks = parsed_by_dir[src].get(strat, [])
                if not blocks:
                    print(f"  WARNING: no sample text found for '{strat}' "
                          f"({ckpt}) in {src} -- merged samples file will be "
                          f"missing this strategy's text.")
                for b in blocks:
                    f.write(b + "\n\n")
        print(f"  wrote {out_path}")

    # ---- merged metrics_<ckpt>.csv (same column logic as eval_checkpoint) ----
    skip_cols = {f"{k}_vals" for k in SCALARS}
    for ckpt, rows in all_results.items():
        keys = []
        for r in rows:
            for k in r:
                if k not in skip_cols and k not in keys:
                    keys.append(k)
        with open(os.path.join(args.out_dir, f"metrics_{ckpt}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows([{k: r.get(k, "") for k in keys} for r in rows])

    # ---- derived artifacts, recomputed from the merged rows via
    # sampling_eval.py's own functions, not reimplemented here ----
    write_pareto(all_results, args.out_dir)
    write_loop_robustness(all_results, args.out_dir)
    write_survival_curves(all_results, args.out_dir, args.n_chars)
    write_method_vs_method(all_results, args.out_dir, args.n_chars)

    with open(os.path.join(args.out_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nMerged {len(args.runs)} run(s) -> {args.out_dir}/ "
          f"(none of the input directories were modified)")


if __name__ == "__main__":
    main()
