"""
validate_loop_event_subword.py

Token-level analogue of validate_loop_event_v2.py's calibrate/freeze/
validate protocol, for the pretrained subword model (GPT-2) pilot flagged
in the paper as still needed before any full decoder comparison there.

WHY THIS IS A SEPARATE SCRIPT, NOT A REUSE OF THE CHARACTER-LEVEL EVENT:
persistent_loop_onset's whitespace-normalization step (collapsing irregular
newline/space insertion before periodicity matching -- see
validate_loop_event_v2.py's docstring for why that was needed) is
string-specific (uses re.sub) and does not apply to a list of token IDs at
all; more fundamentally, a repeated PHRASE in a ~50k-vocabulary subword
tokenizer is naturally a handful of tokens (GPT-2's BPE merges common
multi-character sequences into single tokens), not tens of characters, so
reusing the character-level R=5/period_max=60 setting would be exactly the
kind of untested assumption this project has already been burned by once.

Reuses the SAME calibration and validation corpora as the character-level
protocol (same books, same disjoint calibration/validation split), just
tokenized with the target model's own tokenizer instead of read as raw
characters, so the two events are calibrated on comparably-diverse text.

Usage
-----
  python3 validate_loop_event_subword.py --model gpt2 --out_dir loop_event_subword_report

Then use the frozen (R, period_max) this prints for the full GPT-2 decoder
comparison (Section 11.1 of the paper describes the confirm-only variant of
this workflow for a NEW corpus on an ALREADY-frozen event; this script is
the calibrate-from-scratch step for a genuinely new event, which is what a
new tokenizer/model requires).
"""
import argparse, json, math, os, re, random, hashlib
from urllib.request import urlopen
from urllib.error import URLError, HTTPError


CALIBRATION_BOOKS = {"carroll_alice": 11, "twain_huck": 76, "swift_modest": 1080}
VALIDATION_BOOKS = {
    "melville_moby": 2701, "stoker_dracula": 345, "whitman_leaves": 1322,
    "paine_sense": 147, "homer_iliad": 6130, "doyle_holmes": 1661,
    "dickens_two_cities": 98,
}


def stable_seed(name, salt=0):
    h = hashlib.md5(f"{name}:{salt}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def download_corpus(book_ids_dict, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    texts = {}
    for name, bid in book_ids_dict.items():
        path = os.path.join(cache_dir, f"{bid}.txt")
        if os.path.exists(path):
            texts[name] = open(path, encoding="utf-8", errors="ignore").read()
            continue
        print(f"  downloading {name} (gutenberg #{bid})...")
        candidates = [
            f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt",
            f"https://www.gutenberg.org/files/{bid}/{bid}-0.txt",
            f"https://www.gutenberg.org/files/{bid}/{bid}.txt",
        ]
        raw = None
        for url in candidates:
            try:
                with urlopen(url, timeout=30) as r:
                    raw = r.read()
                break
            except (HTTPError, URLError):
                continue
        if raw is None:
            print(f"    FAILED to download {name}, skipping")
            continue
        try:
            txt = raw.decode("utf-8")
        except UnicodeDecodeError:
            txt = raw.decode("latin-1")
        open(path, "w", encoding="utf-8").write(txt)
        texts[name] = txt
    return texts


def token_windows_of(token_ids, window_tokens, n_windows, seed=0):
    if len(token_ids) < window_tokens:
        return [], []
    rng = random.Random(seed)
    max_start = len(token_ids) - window_tokens
    n_avail = max_start // window_tokens
    n_windows = min(n_windows, max(1, n_avail))
    starts = sorted(rng.sample(range(0, max(1, n_avail)), n_windows)) if n_avail > 0 else [0]
    return [token_ids[s*window_tokens:(s+1)*window_tokens] for s in starts], starts


def persistent_loop_onset_tokens(token_ids, period_max, R, min_p=1,
                                 catch_period_one=True, period_one_min_run=6):
    """
    Same core logic as the character-level persistent_loop_onset, WITHOUT
    the whitespace-normalization step (not applicable to token IDs -- see
    module docstring), and with min_p=1 (a period-1 "loop" at the token
    level is a single token repeated R times, e.g. an emitted token stuck
    repeating, which IS a real and meaningful degenerate case here, unlike
    the character-level case where period-1 needed a separate, higher
    threshold specifically to avoid typesetting dividers -- a concern that
    does not arise in freshly generated token sequences).
    """
    if catch_period_one:
        run = 1
        for i in range(1, len(token_ids)):
            if token_ids[i] == token_ids[i-1]:
                run += 1
                if run >= period_one_min_run:
                    return i - period_one_min_run + 2
            else:
                run = 1
    n = len(token_ids)
    for e in range(min_p * R, n + 1):
        for P in range(min_p, min(period_max, e // R) + 1):
            span = token_ids[e - P:e]
            ok = True
            for r in range(2, R + 1):
                if token_ids[e - r*P:e - (r-1)*P] != span:
                    ok = False; break
            if ok:
                return e
    return -1


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return float("nan"), float("nan")
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out_dir", default="loop_event_subword_report")
    ap.add_argument("--window_tokens_options", nargs="*", type=int,
                     default=[75, 150, 300, 600],
                     help="Token-count analogue of the character-level "
                          "protocol's [300,500,1000,2000] char windows, "
                          "scaled by GPT-2's ~4 chars/token average.")
    ap.add_argument("--period_max_options", nargs="*", type=int, default=[10, 20],
                     help="Much smaller than the character-level 30/60: a "
                          "repeated phrase is naturally a handful of BPE "
                          "tokens, not tens of them.")
    ap.add_argument("--n_windows", type=int, default=200)
    ap.add_argument("--cache_dir", default="gutenberg_cache")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    os.makedirs(args.out_dir, exist_ok=True)
    out = {"model": args.model}

    print("Downloading calibration corpora...")
    calib_texts = download_corpus(CALIBRATION_BOOKS, args.cache_dir)
    print(f"Tokenizing with {args.model}'s tokenizer...")
    calib_tokens = {name: tok.encode(text) for name, text in calib_texts.items()}
    for name, ids in calib_tokens.items():
        print(f"  {name}: {len(ids)} tokens")

    print("\n=== SECTION 1: CALIBRATION (token-level, never re-used after this) ===")
    grid_results = []
    windows_by_wt = {}
    window_indices_record = {}
    for wt in args.window_tokens_options:
        windows_by_wt[wt] = {}
        for name, ids in calib_tokens.items():
            wins, starts = token_windows_of(ids, wt, args.n_windows,
                                            seed=stable_seed(name, salt=wt))
            windows_by_wt[wt][name] = wins
            window_indices_record[f"{name}@wt{wt}"] = starts

    for period_max in args.period_max_options:
        for R in [1, 2, 3, 4, 5, 6]:
            per_window = {}
            worst_ci_hi = 0.0
            for wt in args.window_tokens_options:
                total_fires, total_n = 0, 0
                for name, wins in windows_by_wt[wt].items():
                    total_fires += sum(1 for w in wins
                                       if persistent_loop_onset_tokens(w, period_max, R) >= 0)
                    total_n += len(wins)
                lo, hi = clopper_pearson(total_fires, total_n)
                per_window[wt] = {"fires": total_fires, "n": total_n,
                                   "rate": total_fires/total_n if total_n else float("nan"),
                                   "ci_lo": lo, "ci_hi": hi}
                worst_ci_hi = max(worst_ci_hi, hi)
            grid_results.append({"period_max": period_max, "R": R,
                                  "worst_ci_hi": worst_ci_hi, "per_window": per_window})
            cells = "  ".join(f"w{wt}:{per_window[wt]['fires']}/{per_window[wt]['n']}"
                              f"({per_window[wt]['rate']*100:.1f}%)"
                              for wt in args.window_tokens_options)
            print(f"  period_max={period_max:>3} R={R}: {cells}  "
                  f"worst-case upper CI={worst_ci_hi*100:.2f}%")

    TOLERANCE = 0.01
    frozen = None
    for g in sorted(grid_results, key=lambda x: (x["R"], -x["period_max"])):
        if g["worst_ci_hi"] <= TOLERANCE:
            frozen = g
            break
    met_tolerance = frozen is not None
    if frozen is None:
        frozen = min(grid_results, key=lambda x: (x["worst_ci_hi"], -x["period_max"]))
        print(f"\n  WARNING: no (R, period_max) cell met the {TOLERANCE*100:.0f}% tolerance "
              f"at every window length. Falling back to the lowest worst-case upper CI.")

    R, period_max = frozen["R"], frozen["period_max"]
    print(f"\n  FROZEN (token-level): R={R}, period_max={period_max} tokens "
          f"(worst-case upper 95% CI = {frozen['worst_ci_hi']*100:.2f}%, "
          f"{'met' if met_tolerance else 'did NOT meet'} the 1% tolerance)")

    out["calibration_grid"] = grid_results
    out["calibration_window_indices"] = window_indices_record
    out["frozen"] = {"R": R, "period_max": period_max, "met_tolerance": met_tolerance}

    print("\nDownloading validation battery (disjoint from calibration)...")
    val_texts = download_corpus(VALIDATION_BOOKS, args.cache_dir)
    val_tokens = {name: tok.encode(text) for name, text in val_texts.items()}

    print(f"\n=== SECTION 2: VALIDATION (frozen R={R}, period_max={period_max}) ===")
    rows = []
    total_fires, total_n = 0, 0
    val_window_indices = {}
    for name, ids in val_tokens.items():
        wins, starts = token_windows_of(ids, 150, args.n_windows,
                                        seed=stable_seed(name, salt="validation"))
        val_window_indices[name] = starts
        if not wins:
            continue
        fires = sum(1 for w in wins if persistent_loop_onset_tokens(w, period_max, R) >= 0)
        n = len(wins)
        lo, hi = clopper_pearson(fires, n)
        rows.append({"corpus": name, "fires": fires, "n": n, "rate": fires/n,
                     "ci_lo": lo, "ci_hi": hi})
        total_fires += fires; total_n += n
        print(f"  {name:<20} {fires}/{n}  fire ({fires/n*100:5.1f}%, "
              f"95% CI [{lo*100:.1f}%, {hi*100:.1f}%])")
    pooled_lo, pooled_hi = clopper_pearson(total_fires, total_n)
    print(f"  {'POOLED':<20} {total_fires}/{total_n} fire "
          f"({total_fires/total_n*100:5.1f}%, 95% CI [{pooled_lo*100:.1f}%, {pooled_hi*100:.1f}%])")
    out["validation_battery"] = rows
    out["validation_window_indices"] = val_window_indices
    out["validation_pooled"] = {"fires": total_fires, "n": total_n,
                                 "rate": total_fires/total_n if total_n else float("nan"),
                                 "ci_lo": pooled_lo, "ci_hi": pooled_hi}

    print("\n=== SECTION 3: synthetic-control SENSITIVITY (token-level) ===")
    rng = random.Random(0)
    sens = {}
    for r_true in [2, 3, 4, 5, 8, 10]:
        correct = 0
        n_trials = 30
        for _ in range(n_trials):
            period = rng.choice([1, 2, 3, 5, 8])
            unit = [rng.randint(1000, 40000) for _ in range(period)]
            filler = [rng.randint(1000, 40000) for _ in range(rng.randint(30, 60))]
            seq = filler + unit * r_true + [99999]
            onset = persistent_loop_onset_tokens(seq, period_max, R)
            expected_fire = (r_true >= R)
            if (onset >= 0) == expected_fire:
                correct += 1
        rate = correct / n_trials
        sens[r_true] = rate
        print(f"  constructed repeats={r_true}: {correct}/{n_trials} behaved as expected ({rate*100:.0f}%)")
    out["synthetic_sensitivity"] = sens

    out_path = os.path.join(args.out_dir, "loop_event_subword_report.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull report -> {out_path}")
    print(f"\nFROZEN TOKEN-LEVEL EVENT for {args.model}: persistent cycle, "
          f"R={R}, period<=~{period_max} tokens.")
    print("Use this (R, period_max) pair for the full GPT-2 decoder comparison.")


if __name__ == "__main__":
    main()