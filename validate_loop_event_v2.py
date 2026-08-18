"""
validate_loop_event_v2.py

Addresses two review points about the loop-event validation:

  (3) The persistent-loop threshold (R=3) was picked after looking at BOTH
      Doyle (val) and Dickens (test). Since Dickens is called the test
      corpus, that is circular: the corpus used to validate the event is the
      same corpus the event was tuned on. This script separates CALIBRATION
      corpora (used only to explore R, window length, and period range) from
      a disjoint, never-touched-during-calibration VALIDATION battery, and
      freezes the event definition before looking at the validation battery
      even once.

  (4) "0% false-positive rate" is not a real number without a sample size and
      a confidence interval, and two books is not enough evidence for any
      genre or period claim. This script reports exact Clopper-Pearson
      intervals for every corpus, sweeps window length / repeat count R /
      period range as a small grid (on the calibration set only), and adds
      synthetic degenerate controls -- programmatically constructed text with
      an exactly known period-P, repeat-R cycle -- to check the event's
      SENSITIVITY (does it actually fire on genuine degenerate text) as well
      as its specificity (does it stay quiet on real prose, including a
      poetry corpus with legitimate anaphora/refrain, the hardest case for
      false positives).

Protocol
--------
1. CALIBRATE on a small, fixed set of books never used again in this script.
   Explore R in {1,2,3,4}, window sizes, and period ranges; report the grid
   so the choice is auditable, then FREEZE one setting.
2. VALIDATE the frozen setting, unchanged, on a disjoint battery of books
   spanning multiple authors, genres, and periods (including poetry, for the
   legitimate-repetition stress test), plus Doyle and Dickens now used
   honestly as held-out checks rather than as the source of the choice.
3. SENSITIVITY: run the frozen event against synthetic text with known
   inserted cycles at many periods and repeat counts, to confirm it detects
   genuine degeneracy at the frozen threshold.

This script downloads books from Project Gutenberg via urllib, so it must be
run on a machine with internet access (it does not run inside a network-
restricted sandbox). It does not need the trained model or any generated
text for Sections 1-3; Section 4 additionally checks the frozen event against
this project's own generated samples if --samples is given.

Usage
-----
  python3 validate_loop_event_v2.py --out_dir loop_event_v2_report
  python3 validate_loop_event_v2.py --out_dir loop_event_v2_report \
      --samples runs/v8_loop_regime/samples_cosine.txt --ckpt_name cosine
"""

import os, re, json, argparse, random, hashlib
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

import numpy as np

try:
    from scipy.stats import beta as _beta
    def clopper_pearson(k, n, alpha=0.05):
        """Exact binomial CI. k successes out of n trials."""
        lo = 0.0 if k == 0 else _beta.ppf(alpha/2, k, n-k+1)
        hi = 1.0 if k == n else _beta.ppf(1-alpha/2, k+1, n-k)
        return float(lo), float(hi)
except Exception:
    def clopper_pearson(k, n, alpha=0.05):
        """Wilson-interval fallback if scipy is unavailable (approximate)."""
        if n == 0:
            return (0.0, 1.0)
        z = 1.96
        p = k / n
        denom = 1 + z*z/n
        centre = p + z*z/(2*n)
        half = z * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5)
        return (max(0.0, (centre - half)/denom), min(1.0, (centre + half)/denom))


START_RE = re.compile(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
END_RE   = re.compile(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)


def fetch_gutenberg_text(book_id):
    candidates = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]
    last_err = None
    for url in candidates:
        try:
            with urlopen(url, timeout=30) as r:
                raw = r.read()
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("latin-1")
        except (HTTPError, URLError) as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not download book_id={book_id}: {last_err}")


def strip_boilerplate(text):
    m1, m2 = START_RE.search(text), END_RE.search(text)
    if m1 and m2 and m2.start() > m1.end():
        return text[m1.end():m2.start()]
    lines = text.splitlines()
    return "\n".join(lines[50:-50]) if len(lines) > 200 else text


def normalize(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(l.rstrip() for l in text.splitlines()).strip()


# CALIBRATION set: used ONLY in Section 1, never referenced again after the
# threshold is frozen. Deliberately disjoint from train/val/test authors
# (Austen, Shelley, Doyle, Dickens) and from the VALIDATION battery below.
CALIBRATION_BOOKS = {
    "carroll_alice":  11,     # Lewis Carroll, fantasy, Victorian
    "twain_huck":     76,     # Mark Twain, American vernacular, dialect-heavy
    "swift_modest":   1080,   # Jonathan Swift, satirical essay, 18th c.
}

# VALIDATION battery: disjoint from calibration, spans authors/genres/periods
# not touched while choosing the threshold. Includes a POETRY corpus
# (Whitman) specifically because refrain and anaphora are the most likely
# source of legitimate short repeats that could produce a false positive.
VALIDATION_BOOKS = {
    "melville_moby":   2701,  # American novel, long philosophical asides
    "stoker_dracula":  345,   # epistolary gothic novel
    "whitman_leaves":  1322,  # POETRY -- anaphora / refrain stress test
    "paine_sense":     147,   # political essay, 18th c. non-fiction
    "homer_iliad":     6130,  # translated epic poetry
    # Honest re-use, now as genuine held-out checks rather than the source of
    # the choice:
    "doyle_holmes":    1661,
    "dickens_two_cities": 98,
}


def _normalize_whitespace_for_loop_check(text):
    """Collapse whitespace runs to a single space before periodicity matching.
    See persistent_loop_onset docstring for why this is necessary."""
    return re.sub(r"\s+", " ", text)


def persistent_loop_onset(text, period_max, R, min_p=2, catch_period_one=True,
                           period_one_min_run=8):
    """
    period-1 loops (a single character repeated many times, e.g. "aaaaaa")
    are, by construction, invisible to the P>=min_p=2 search unless the run
    happens to be long enough to also satisfy a period-2 check (needs
    2*R characters, not R). catch_period_one adds an explicit, separate
    check for a run of period_one_min_run or more identical characters.

    period_one_min_run is DECOUPLED from R deliberately: plain-text book
    transcriptions commonly use short runs of identical characters as
    typesetting -- scene-break dividers ("-----", "* * * * *" without the
    spaces, "_____") -- typically 3-10 characters long, and using R itself
    as the period-1 threshold made the detector trip on these at R=5 (a
    5-dash divider exactly matches run>=R=5). period_one_min_run=8 sits
    above ordinary divider lengths while remaining far below the length of
    genuine degenerate model output, which was verified separately.
    """
    text = _normalize_whitespace_for_loop_check(text)

    if catch_period_one:
        run = 1
        for i in range(1, len(text)):
            if text[i] == text[i-1]:
                run += 1
                if run >= period_one_min_run:
                    return i - period_one_min_run + 2
            else:
                run = 1
    n = len(text)
    for e in range(min_p * R, n + 1):
        for P in range(min_p, min(period_max, e // R) + 1):
            span = text[e - P:e]
            ok = True
            for r in range(2, R + 1):
                if text[e - r*P:e - (r-1)*P] != span:
                    ok = False; break
            if ok:
                return e
    return -1


def stable_seed(name, salt=0):
    """
    Deterministic, process-independent seed from a string. Python's builtin
    hash() is randomized per-process (PYTHONHASHSEED) unless explicitly
    disabled, so seed=hash(name) silently produced a DIFFERENT sample of
    windows on every run -- this is why numbers in a saved report could
    disagree with a later re-run on the same corpus. hashlib.md5 is stable
    across processes and Python versions.
    """
    h = hashlib.md5(f"{name}:{salt}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def windows_of(text, window_chars, n_windows, seed=0):
    """Returns (window_texts, window_start_indices) so the exact sampled
    windows are reconstructable and auditable, not just their count."""
    if len(text) < window_chars:
        return [], []
    rng = random.Random(seed)
    max_start = len(text) - window_chars
    n_avail = max_start // window_chars
    n_windows = min(n_windows, max(1, n_avail))
    starts = sorted(rng.sample(range(0, max(1, n_avail)), n_windows)) if n_avail > 0 else [0]
    texts = [text[s*window_chars:(s+1)*window_chars] for s in starts]
    return texts, starts


def download_corpus(book_ids_dict, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    texts = {}
    for name, bid in book_ids_dict.items():
        cache_path = os.path.join(cache_dir, f"{bid}.txt")
        if os.path.exists(cache_path):
            texts[name] = open(cache_path, encoding="utf-8").read()
            continue
        print(f"  downloading {name} (gutenberg #{bid})...")
        raw = fetch_gutenberg_text(bid)
        clean = normalize(strip_boilerplate(raw))
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(clean)
        texts[name] = clean
    return texts


def make_synthetic_degenerate(period, repeats, filler_len=200, vocab="abcdefghijklmnopqrstuvwxyz ", seed=0):
    """Random filler, then an EXACT period-P cycle repeated `repeats` times."""
    rng = random.Random(seed)
    filler = "".join(rng.choice(vocab) for _ in range(filler_len))
    cycle_unit = "".join(rng.choice(vocab) for _ in range(period))
    return filler + cycle_unit * repeats


def run_calibration_grid(calib_texts, window_chars_options, n_windows, out):
    """
    Explore R in {1..6} x period_max in {30,60} x SEVERAL window lengths, all
    on the CALIBRATION corpora only. Freezing on a single window length and
    then discovering (via the validation battery) that the choice is not
    stable at other window lengths would mean the final R was effectively
    picked by looking at validation data -- exactly the contamination this
    protocol exists to avoid. So window-length robustness is decided HERE,
    calibration-only, before validation is touched even once.
    """
    print("\n=== SECTION 1: CALIBRATION (never re-used after this) ===")
    print(f"  window lengths tested: {window_chars_options}")
    grid_results = []
    period_max_options = [30, 60]
    R_options = [1, 2, 3, 4, 5, 6]

    windows_by_wc = {}
    window_indices_record = {}   # saved into the report for exact reproducibility auditing
    for wc in window_chars_options:
        windows_by_wc[wc] = {}
        for name, text in calib_texts.items():
            texts, starts = windows_of(text, wc, n_windows, seed=stable_seed(name, salt=wc))
            windows_by_wc[wc][name] = texts
            window_indices_record[f"{name}@wc{wc}"] = starts

    for period_max in period_max_options:
        for R in R_options:
            # worst-case (max) upper CI across all tested window lengths --
            # this is what "safe at every window length we plan to use" means.
            per_window = {}
            worst_ci_hi = 0.0
            for wc in window_chars_options:
                total_fires, total_n = 0, 0
                for name, wins in windows_by_wc[wc].items():
                    total_fires += sum(1 for w in wins if persistent_loop_onset(w, period_max, R) >= 0)
                    total_n += len(wins)
                lo, hi = clopper_pearson(total_fires, total_n)
                per_window[wc] = {"fires": total_fires, "n": total_n, "rate": total_fires/total_n if total_n else float("nan"),
                                   "ci_lo": lo, "ci_hi": hi}
                worst_ci_hi = max(worst_ci_hi, hi)
            grid_results.append({"period_max": period_max, "R": R,
                                  "worst_ci_hi": worst_ci_hi, "per_window": per_window})
            cells = "  ".join(f"w{wc}:{per_window[wc]['fires']}/{per_window[wc]['n']}({per_window[wc]['rate']*100:.1f}%)"
                              for wc in window_chars_options)
            print(f"  period_max={period_max:>3} R={R}: {cells}  worst-case upper CI={worst_ci_hi*100:.2f}%")

    TOLERANCE = 0.01
    frozen = None
    met_tolerance = False
    for g in sorted(grid_results, key=lambda x: (x["R"], -x["period_max"])):
        if g["worst_ci_hi"] <= TOLERANCE:
            frozen = g; met_tolerance = True
            break
    if frozen is None:
        frozen = min(grid_results, key=lambda x: (x["worst_ci_hi"], -x["period_max"]))
        print(f"\n  WARNING: no (R, period_max) cell met the {TOLERANCE*100:.0f}% tolerance "
              f"at EVERY tested window length. Falling back to the cell with the lowest "
              f"worst-case upper CI across window lengths.")
    tag = f"meets the {TOLERANCE*100:.0f}% tolerance at every window length" if met_tolerance else \
          f"does NOT meet the {TOLERANCE*100:.0f}% tolerance at every window length (fallback)"
    print(f"\n  FROZEN: R={frozen['R']}, period_max={frozen['period_max']} "
          f"(worst-case upper 95% CI across window lengths = {frozen['worst_ci_hi']*100:.2f}%, {tag})")
    out["calibration_grid"] = grid_results
    out["calibration_window_indices"] = window_indices_record
    out["frozen"] = {"R": frozen["R"], "period_max": frozen["period_max"],
                      "tolerance": TOLERANCE, "met_tolerance": met_tolerance,
                      "worst_case_ci_hi": frozen["worst_ci_hi"]}
    return frozen["R"], frozen["period_max"]


def run_validation_battery(val_texts, R, period_max, window_chars, n_windows, out):
    print(f"\n=== SECTION 2: VALIDATION (frozen R={R}, period_max={period_max}), "
          f"never touched during calibration ===")
    rows = []
    val_window_indices = {}
    for name, text in val_texts.items():
        wins, starts = windows_of(text, window_chars, n_windows, seed=stable_seed(name, salt="validation"))
        val_window_indices[name] = starts
        if not wins:
            continue
        fires = sum(1 for w in wins if persistent_loop_onset(w, period_max, R) >= 0)
        lo, hi = clopper_pearson(fires, len(wins))
        rows.append({"corpus": name, "fires": fires, "n": len(wins),
                      "rate": fires/len(wins), "ci_lo": lo, "ci_hi": hi})
        print(f"  {name:<20} {fires:>3}/{len(wins):<4} fire "
              f"({fires/len(wins)*100:5.1f}%, 95% CI [{lo*100:.1f}%, {hi*100:.1f}%])")
    total_fires = sum(r["fires"] for r in rows)
    total_n = sum(r["n"] for r in rows)
    lo, hi = clopper_pearson(total_fires, total_n)
    print(f"  {'POOLED':<20} {total_fires:>3}/{total_n:<4} fire "
          f"({total_fires/total_n*100:5.1f}%, 95% CI [{lo*100:.1f}%, {hi*100:.1f}%])")
    out["validation_battery"] = rows
    out["validation_window_indices"] = val_window_indices
    out["validation_pooled"] = {"fires": total_fires, "n": total_n,
                                 "rate": total_fires/total_n,
                                 "ci_lo": lo, "ci_hi": hi}


def run_window_length_robustness_confirmatory(val_texts, R, period_max, out):
    """
    CONFIRMATORY ONLY: R and period_max were already frozen in Section 1
    using calibration data across multiple window lengths. This section does
    not choose anything -- it just reports how the already-frozen setting
    behaves on the validation battery at different window lengths, so a
    discrepancy is visible rather than silently used to re-tune R.
    """
    print(f"\n=== SECTION 3: window-length behaviour of the FROZEN setting "
          f"(R={R}, period_max={period_max}) on the validation battery ===")
    print("  (confirmatory only -- R was already frozen in Section 1 from "
          "calibration data alone)")
    robustness_indices = {}
    for window_chars in (300, 500, 1000, 2000):
        fires, n = 0, 0
        for name, text in val_texts.items():
            wins, starts = windows_of(text, window_chars, n_windows=30,
                               seed=stable_seed(name, salt=f"robustness{window_chars}"))
            robustness_indices[f"{name}@wc{window_chars}"] = starts
            fires += sum(1 for w in wins if persistent_loop_onset(w, period_max, R) >= 0)
            n += len(wins)
        if n == 0:
            continue
        lo, hi = clopper_pearson(fires, n)
        print(f"  window={window_chars:>5}: {fires}/{n} fire "
              f"({fires/n*100:.1f}%, 95% CI [{lo*100:.1f}%, {hi*100:.1f}%])")
        out.setdefault("window_robustness_confirmatory", []).append(
            {"window_chars": window_chars, "fires": fires, "n": n,
             "rate": fires/n, "ci_lo": lo, "ci_hi": hi})
    out["window_robustness_indices"] = robustness_indices


def run_synthetic_sensitivity(R, period_max, out, period_one_min_run=8):
    print(f"\n=== SECTION 4: synthetic-control SENSITIVITY (frozen R={R}) ===")
    periods = [1, 2, 3, 5, 8, 13, 20, 30, 45, 60]
    repeats_grid = [2, 3, 4, 5, 8, 10]
    rows = []
    for reps in repeats_grid:
        detected, total = 0, 0
        for p in periods:
            for seed in range(5):
                text = make_synthetic_degenerate(p, reps, seed=seed)
                onset = persistent_loop_onset(text, period_max, R,
                                               period_one_min_run=period_one_min_run)
                # For period=1, the actual detector uses period_one_min_run,
                # not R, as the threshold (see persistent_loop_onset). For
                # every other period, R is still the correct threshold.
                threshold = period_one_min_run if p == 1 else R
                should_detect = reps >= threshold
                fired = onset >= 0
                total += 1
                if should_detect == fired:
                    detected += 1
        rate = detected/total
        rows.append({"repeats_constructed": reps, "correct": detected, "total": total, "rate": rate})
        print(f"  constructed repeats={reps}: {detected}/{total} behaved as expected ({rate*100:.0f}%)")
    out["synthetic_sensitivity"] = rows


def run_on_generated_samples(samples_path, ckpt_name, R, period_max, out):
    if not samples_path or not os.path.exists(samples_path):
        return
    print(f"\n=== SECTION 5: frozen event on this project's generated samples ===")
    txt = open(samples_path, encoding="utf-8").read()
    methods = sorted(set(re.findall(r"\[([a-z0-9_.]+)\]\[" + re.escape(ckpt_name) + r"\] prompt=", txt)))
    rows = []
    for m in methods:
        i, samples = 0, []
        while True:
            i = txt.find(f"[{m}][{ckpt_name}] prompt=", i)
            if i < 0:
                break
            lines = txt[i:].split("\n"); body = []
            for l in lines[2:]:
                if l.strip() == "" and body:
                    break
                body.append(l)
            samples.append(" ".join(body).strip()); i += 10
        if not samples:
            continue
        fires = sum(1 for s in samples if persistent_loop_onset(s, period_max, R) >= 0)
        rows.append({"method": m, "fires": fires, "n": len(samples)})
        print(f"  {m:<20} {fires}/{len(samples)} fire under the frozen event")
    out["generated_samples_check"] = rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="loop_event_v2_report")
    ap.add_argument("--window_chars", type=int, default=500,
                     help="Primary window length, used for the Section 2 "
                          "validation-battery report.")
    ap.add_argument("--window_chars_options", nargs="*", type=int,
                     default=[300, 500, 1000, 2000],
                     help="Window lengths tested DURING calibration (Section 1), "
                          "so window-length robustness is decided before "
                          "validation data is touched at all, not discovered "
                          "afterward.")
    ap.add_argument("--n_windows", type=int, default=300)
    ap.add_argument("--cache_dir", default="gutenberg_cache")
    ap.add_argument("--samples", default=None)
    ap.add_argument("--ckpt_name", default="cosine")
    ap.add_argument("--confirm_only", nargs="*", type=int, default=None,
                     help="Skip calibration entirely and reuse the ALREADY-"
                          "FROZEN (R=5, period_max=60) setting; run only a "
                          "confirmatory false-positive check on the given "
                          "Gutenberg book IDs. For generalization to a new "
                          "corpus/genre (e.g. non-fiction), this is the "
                          "correct mode: it does not re-tune the threshold "
                          "on the new corpus (which would be circular), it "
                          "only checks whether the existing frozen event "
                          "remains valid there. Pass the new corpus's OWN "
                          "val+test book IDs, e.g. "
                          "--confirm_only 205 1232 for a non-fiction check.")
    ap.add_argument("--confirm_R", type=int, default=5)
    ap.add_argument("--confirm_period_max", type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out = {}

    if args.confirm_only is not None:
        print(f"CONFIRM-ONLY mode: reusing frozen R={args.confirm_R}, "
              f"period_max={args.confirm_period_max} (no re-calibration). "
              f"Checking false-positive rate on book IDs {args.confirm_only}.")
        new_corpus = {f"book_{bid}": bid for bid in args.confirm_only}
        new_texts = download_corpus(new_corpus, args.cache_dir)
        run_validation_battery(new_texts, args.confirm_R, args.confirm_period_max,
                                args.window_chars, args.n_windows, out)
        run_synthetic_sensitivity(args.confirm_R, args.confirm_period_max, out)
        run_on_generated_samples(args.samples, args.ckpt_name,
                                  args.confirm_R, args.confirm_period_max, out)
        out_path = os.path.join(args.out_dir, "loop_event_confirm_report.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nConfirmatory report -> {out_path}")
        print("If the pooled false-positive rate here is comparable to the "
              "original validation battery's ~1%, the frozen event is "
              "confirmed valid on this new corpus and can be used as-is for "
              "the generalization experiment without re-running calibration.")
        return

    print("Downloading calibration corpora...")
    calib_texts = download_corpus(CALIBRATION_BOOKS, args.cache_dir)
    R, period_max = run_calibration_grid(calib_texts, args.window_chars_options,
                                          args.n_windows, out)

    print("\nDownloading validation battery (disjoint from calibration)...")
    val_texts = download_corpus(VALIDATION_BOOKS, args.cache_dir)
    run_validation_battery(val_texts, R, period_max, args.window_chars, args.n_windows, out)
    run_window_length_robustness_confirmatory(val_texts, R, period_max, out)
    run_synthetic_sensitivity(R, period_max, out)
    run_on_generated_samples(args.samples, args.ckpt_name, R, period_max, out)

    out_path = os.path.join(args.out_dir, "loop_event_v2_report.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull report -> {out_path}")
    print(f"\nFROZEN EVENT: persistent cycle, R={R}, period<=~{period_max} chars.")
    print("Use this (R, period_max) pair in sampling_eval.py / decoding.py consistently.")


if __name__ == "__main__":
    main()