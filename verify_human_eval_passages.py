"""
verify_human_eval_passages.py

Identifies which saved generation run each passage shown in
human_eval_tool.html came from, by matching the passage text against the
samples_cosine.txt files of three runs:

  v3_final   the first evaluated run (original FSD-style implementation)
  v6         FSD-style with fixed normalized interpolation weights beta^(N-k)
             and the final LZ Penalty port
  v7         FSD-style with the official recursive residual-mass anti-LM
             (the version reported in the paper's tables)

Usage:  python3 verify_human_eval_passages.py
"""
import json, re
from collections import Counter

RUNS = {
    "v3_final": "runs/main_comparison_v3_final/samples_cosine.txt",
    "v6":       "runs/main_comparison_v6_fsdlz_merged/samples_cosine.txt",
    "v7":       "runs/main_comparison_v7_fsd2_merged/samples_cosine.txt",
}
STRATEGY = {"dual": "lt_dual_eps0.05", "suffixmatch": "lt_suffixmatch_a3.0",
            "fsd": "lt_fsd", "lzpenalty": "lt_lzpenalty"}
HDR = re.compile(r"^\[([^\]]+)\]\[cosine\] prompt='.*?' seed=\d+\n-{60}\n", re.M)


def parse(path):
    txt = open(path).read()
    ms = list(HDR.finditer(txt))
    out = {}
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(txt)
        out.setdefault(m.group(1), []).append(txt[m.end():end].strip())
    return out


def main():
    html = open("human_eval_tool.html").read()
    blob = re.search(r'<script id="human-eval-data" type="application/json">(.*?)</script>',
                     html, re.S).group(1)
    pairs = json.loads(blob)["task2"]
    runs = {k: parse(v) for k, v in RUNS.items()}
    found = Counter()
    for pair in pairs:
        for side in ("a", "b"):
            method, text = pair["method_" + side], pair["text_" + side].strip()
            for run, parsed in runs.items():
                hit = any(t.startswith(text[:80]) for t in parsed.get(STRATEGY[method], []))
                found[(method, run, hit)] += 1
    print(f"{'method':<12}{'run':<10}{'passages found':>16}")
    for method in STRATEGY:
        for run in RUNS:
            print(f"{method:<12}{run:<10}{found[(method, run, True)]:>8} / "
                  f"{found[(method, run, True)] + found[(method, run, False)]}")


if __name__ == "__main__":
    main()
