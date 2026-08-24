"""
gpt2_full_comparison.py

Full decoder comparison on a pretrained SUBWORD model (GPT-2 by default),
run only AFTER pretrained_subword_pilot.py has confirmed a measurable
low-temperature repetition regime exists and validate_loop_event_subword.py
has frozen this model's OWN (R, period_max) event -- reusing the
character-level thresholds here would repeat exactly the mistake already
found and fixed once in this project (see those scripts' docstrings).

Deliberately reuses, rather than reimplements:
  - The decoder classes themselves (SuffixMatchDecoder, FSDDecoder,
    LZPenaltyDecoder, RecurrenceRiskDecoder) from src/decoding.py. Every one
    of these operates only on a logits vector and a list of generated token
    ids -- nothing in their .step()/.reset()/.prime()/.diagnostics()
    interface is character-model-specific, so the SAME classes that were
    fixed (log-space dual projection, renamed SuffixMatchDecoder, real LZ
    Penalty) are exercised here verbatim, not re-derived and risking a
    second, divergent implementation of the same method.
  - The generic (model-agnostic) metric and inference functions from
    sampling_eval.py: prompt-clustered bootstrap CIs, Kaplan-Meier survival/
    RMST/loop-rate, and the text-only quality metrics (compression ratio,
    n-gram entropy/mass, longest repeated substring, spelling error rate).
    None of these read anything from the character model; duplicating them
    here would risk the two comparisons silently diverging in definition.

What is NOT reused: sampling_eval.py's own generate() and
model_consistency_nll(), because both assume a CharTransformerLM (fixed
block_size attribute, a from-scratch causal LM with no KV-cache API, no
attention_mask). This module's generate_hf() and mc_nll_bits() are the
subword-model analogues, kept deliberately simple (GPT-2's 1024-token
context comfortably covers prompt + n_tokens in one forward pass, so the
padding-corruption bug class fixed for the character model's mcNLL --
batching variable-length windows -- does not arise here at all: every
sample is scored in its own unbatched, unpadded forward pass).

Usage
-----
  python3 gpt2_full_comparison.py --model gpt2 --n_seeds 10 --n_tokens 200 \
      --out_dir runs/gpt2_full_comparison

Run pretrained_subword_pilot.py and validate_loop_event_subword.py first;
this script reads the frozen (R, period_max) from
loop_event_subword_report/loop_event_subword_report.json and will refuse to
run with placeholder values if that file is missing.
"""
import os, sys, argparse, csv, json, math, time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.decoding import (SuffixMatchDecoder, FSDDecoder, LZPenaltyDecoder,
                          RecurrenceRiskDecoder)
# Reused verbatim -- see module docstring for why these specific functions,
# and not sampling_eval.py's generate()/model_consistency_nll(), are shared.
from sampling_eval import (cluster_bootstrap_ci, kaplan_meier, survival_auc,
                           rmst, loop_rate, compression_ratio,
                           longest_repeated_substring, char_ngram_entropy,
                           spelling_error_rate)


# ---------------------------------------------------------------------------
# PROMPT PROTOCOL, same discipline as sampling_eval.py's DEV/TEST split.
# PILOT_PROMPTS (in pretrained_subword_pilot.py) were used to confirm the
# regime exists; these 10 are written fresh afterwards and never inspected
# while choosing temperature/alpha/eps for this comparison.
# ---------------------------------------------------------------------------
TEST_PROMPTS = [
    ("g_report",   "The report concluded that"),
    ("g_meeting",  "During the meeting, the committee"),
    ("g_history",  "The history of the region begins with"),
    ("g_letter",   "In her letter, she explained that"),
    ("g_weather",  "By the time the storm passed,"),
    ("g_project",  "The engineering team decided to"),
    ("g_market",   "According to the latest figures, the market"),
    ("g_journey",  "Halfway through the journey, they realized"),
    ("g_experiment","The results of the experiment showed that"),
    ("g_city",     "Walking through the old city, one notices"),
]


def load_frozen_event(path="loop_event_subword_report/loop_event_subword_report.json"):
    if not os.path.exists(path):
        raise SystemExit(
            f"Missing {path}. Run validate_loop_event_subword.py first -- this "
            f"script refuses to fall back to a placeholder (R, period_max) "
            f"rather than silently reusing an unvalidated or character-level "
            f"threshold.")
    d = json.load(open(path))
    frozen = d["frozen"]
    if not frozen.get("met_tolerance", False):
        print(f"  WARNING: frozen event did not meet the calibration tolerance "
              f"(worst-case upper CI shown in {path}); proceeding with the "
              f"best-available (R, period_max) anyway, as validate_loop_event_"
              f"subword.py's own fallback does, but flagging it here too.")
    return frozen["R"], frozen["period_max"]


def persistent_loop_onset_tokens(token_ids, R, period_max, min_p=1,
                                 catch_period_one=True, period_one_min_run=6):
    """Identical logic to validate_loop_event_subword.py's frozen event --
    duplicated (not imported) only because that module is a standalone
    calibration script, not a library; kept byte-identical on purpose."""
    if catch_period_one:
        run = 1
        for i in range(1, len(token_ids)):
            if token_ids[i] == token_ids[i - 1]:
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
                if token_ids[e - r * P:e - (r - 1) * P] != span:
                    ok = False; break
            if ok:
                return e
    return -1


def rep_ngram_mass_tokens(ids, n):
    g = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    if not g: return 0.0
    c = Counter(g)
    return sum(v for v in c.values() if v > 1) / len(g)


@torch.no_grad()
def mc_nll_bits(model, full_ids, prompt_len, device):
    """NLL (bits/token) of the generated continuation under the SAME model,
    scored on positions >= prompt_len. GPT-2's 1024-token context covers
    prompt+generation in one unbatched, unpadded forward pass -- see module
    docstring for why the character-model mcNLL's padding fix does not need
    a subword-model analogue here."""
    if len(full_ids) < prompt_len + 2:
        return float("nan")
    x = torch.tensor([full_ids[:-1]], device=device)
    y = torch.tensor([full_ids[1:]], device=device)
    logits = model(x).logits
    lp = F.log_softmax(logits, dim=-1)
    tok_nll = -lp.gather(2, y.unsqueeze(-1)).squeeze(-1)[0]
    gen_nll = tok_nll[prompt_len - 1:]
    if gen_nll.numel() == 0:
        return float("nan")
    return float(gen_nll.mean().item() / math.log(2))


def make_decoder(cfg, vocab_size):
    if cfg.get("risk") is not None:
        return RecurrenceRiskDecoder(**cfg["risk"])
    if cfg.get("fsd") is not None:
        return FSDDecoder(**cfg["fsd"])
    if cfg.get("lz_penalty") is not None:
        return LZPenaltyDecoder(vocab_size=vocab_size, **cfg["lz_penalty"])
    if cfg.get("suffixmatch") is not None:
        return SuffixMatchDecoder(**cfg["suffixmatch"])
    return None


@torch.no_grad()
def generate_hf(model, prompt_ids, n_tokens, device, temperature=1.0,
                rep_penalty=1.0, no_repeat_ngram=0, decoder=None,
                measure_time=False):
    """
    Minimal HF-model generation loop, structurally parallel to
    src/decoding.py's generate() but sourcing logits from an HF causal LM
    forward pass instead of CharTransformerLM. When `decoder` is given, every
    single step is delegated to decoder.step(logits, generated_ids) exactly
    as the character-model generate() does for its adaptive/lz_decoder slot
    -- this is what lets SuffixMatchDecoder/FSDDecoder/LZPenaltyDecoder/
    RecurrenceRiskDecoder run unmodified against a completely different
    model family.
    """
    ids = list(prompt_ids)
    x = torch.tensor([ids], device=device)

    if decoder is not None:
        decoder.reset()
        # SuffixMatchDecoder has no prime() -- unlike RecurrenceRiskDecoder/
        # FSDDecoder/LZPenaltyDecoder, it reads history straight from the
        # generated_ids argument passed to step() every call, so there is
        # nothing to pre-register. Matches src/decoding.py's generate(),
        # which likewise never calls .prime() on its lz_decoder slot.
        if hasattr(decoder, "prime"):
            decoder.prime(ids)

    nr_followers = None
    if no_repeat_ngram > 0 and decoder is None:
        nr_followers = defaultdict(set)
        for i in range(len(ids) - (no_repeat_ngram - 1)):
            ctx = tuple(ids[i:i + no_repeat_ngram - 1])
            nr_followers[ctx].add(ids[i + no_repeat_ngram - 1])

    t0 = time.perf_counter() if measure_time else None
    for _ in range(n_tokens):
        logits = model(x).logits[0, -1, :]

        if decoder is not None:
            tok = decoder.step(logits, ids)
            ids.append(tok)
            x = torch.cat([x, torch.tensor([[tok]], device=device)], dim=1)
            continue

        if temperature == 0.0:
            tok = int(torch.argmax(logits).item())
            ids.append(tok)
            x = torch.cat([x, torch.tensor([[tok]], device=device)], dim=1)
            continue

        logits = logits.clone() / temperature
        if rep_penalty != 1.0:
            for tid in set(ids):
                if logits[tid] > 0: logits[tid] /= rep_penalty
                else: logits[tid] *= rep_penalty
        if no_repeat_ngram > 0 and len(ids) >= no_repeat_ngram - 1:
            ctx = tuple(ids[-(no_repeat_ngram - 1):]) if no_repeat_ngram > 1 else ()
            for tid in nr_followers.get(ctx, ()):
                logits[tid] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        tok = int(torch.multinomial(probs, 1).item())
        ids.append(tok)
        if nr_followers is not None and len(ids) >= no_repeat_ngram:
            nr_followers[tuple(ids[-no_repeat_ngram:-1])].add(ids[-1])
        x = torch.cat([x, torch.tensor([[tok]], device=device)], dim=1)

    cps = None
    if measure_time:
        elapsed = time.perf_counter() - t0
        cps = round(n_tokens / elapsed, 1) if elapsed > 0 else 0.0
    return ids, cps


def build_configs():
    """Mirrors sampling_eval.py's loop-regime config set, at token
    granularity. Temperature held at 0.10 for every non-temperature-sweep
    config, matching the primary character-model comparison's convention
    (same held-T logic: comparing decoders at a T where the pathology is
    absent compares zeros)."""
    C = []
    def add(name, **kw):
        d = dict(temperature=0.10, rep_penalty=1.0, no_repeat_ngram=0,
                 risk=None, fsd=None, lz_penalty=None, suffixmatch=None)
        d.update(kw); d["name"] = name
        C.append(d)

    add("greedy", temperature=0.0)
    for t in (0.05, 0.10, 0.15):
        add(f"lt_temp_{t}", temperature=t)
    add("lt_rep_penalty_1.3", rep_penalty=1.3)
    add("lt_no_repeat_3gram", no_repeat_ngram=3)
    add("lt_suffixmatch_a3.0",
        suffixmatch=dict(temperature=0.10, top_p=1.0, alpha=3.0,
                         max_history=400, ref_len=20))
    add("lt_fsd", fsd=dict(temperature=0.10, top_p=1.0, alpha=4.0, n_min=2, n_max=4))
    add("lt_lzpenalty",
        # buffer/window at the paper's own subword defaults (their 128k-vocab
        # setting), not the char-model's scaled-down 8/128: GPT-2's ~50k
        # vocabulary is far closer to the paper's regime than to a ~70-symbol
        # character vocabulary.
        lz_penalty=dict(temperature=0.10, top_p=1.0, alpha=0.15,
                        buffer_size=32, window_size=512))
    add("lt_risk_only",
        risk=dict(temperature=0.10, top_p=1.0, n_min=2, n_max=4,
                  mode="fixed", alpha_base=3.0, include_prompt_context=True))
    add("lt_adaptive",
        risk=dict(temperature=0.10, top_p=1.0, n_min=2, n_max=4,
                  mode="adaptive", alpha_base=3.0, alpha_max=12.0,
                  lambda_rep=15.0, lambda_ent=1.0, rep_target=0.05,
                  ent_target=4.0, window=64))
    for e in (0.01, 0.05):
        add(f"lt_dual_eps{e}",
            risk=dict(temperature=0.10, top_p=1.0, n_min=2, n_max=4,
                      mode="dual", eps=e))
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--n_tokens", type=int, default=200)
    ap.add_argument("--out_dir", default="runs/gpt2_full_comparison")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--event_report",
                    default="loop_event_subword_report/loop_event_subword_report.json")
    args = ap.parse_args()

    R, period_max = load_frozen_event(args.event_report)
    print(f"Frozen token-level event for {args.model}: R={R}, period_max={period_max}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    device = (torch.device("mps") if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Loading {args.model} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    vocab_size = model.config.vocab_size

    os.makedirs(args.out_dir, exist_ok=True)
    configs = build_configs()
    if args.only:
        configs = [c for c in configs if c["name"] in set(args.only)]

    samples_f = open(os.path.join(args.out_dir, f"samples_{args.model}.txt"),
                     "w", encoding="utf-8")
    rows = []
    for c in configs:
        print(f"  {c['name']:<22}", end="", flush=True)
        onsets, groups = [], []
        acc = {"rep_mass_3": [], "compression": [], "longest_rep_sub": [],
               "mc_nll_bits": [], "spelling_error": [], "entropy_4char": []}
        diag = {"kl_bits": [], "risk_achieved": [], "lambda_mean": [],
                "dual_feasible_rate": [], "dual_structurally_infeasible_rate": []}
        cps_val = None

        for pi, (pname, ptext) in enumerate(TEST_PROMPTS):
            for seed in range(1, args.n_seeds + 1):
                print(f"\r  {c['name']:<24} sample {pi*args.n_seeds+seed}/"
                      f"{len(TEST_PROMPTS)*args.n_seeds}", end="", flush=True)
                torch.manual_seed(seed * 1000 + pi)
                np.random.seed(seed * 1000 + pi)
                prompt_ids = tok(ptext, return_tensors="pt").input_ids[0].tolist()
                decoder = make_decoder(c, vocab_size)
                measure = (pi == 0 and seed == 1)
                full_ids, cps = generate_hf(
                    model, prompt_ids, args.n_tokens, device,
                    temperature=c["temperature"], rep_penalty=c["rep_penalty"],
                    no_repeat_ngram=c["no_repeat_ngram"], decoder=decoder,
                    measure_time=measure)
                if cps is not None:
                    cps_val = cps
                gen_ids = full_ids[len(prompt_ids):]
                text = tok.decode(gen_ids)

                groups.append(pi)
                onsets.append(persistent_loop_onset_tokens(gen_ids, R, period_max))
                acc["rep_mass_3"].append(rep_ngram_mass_tokens(gen_ids, 3))
                acc["compression"].append(compression_ratio(text))
                acc["longest_rep_sub"].append(longest_repeated_substring(text))
                acc["mc_nll_bits"].append(
                    mc_nll_bits(model, full_ids, len(prompt_ids), device))
                acc["spelling_error"].append(spelling_error_rate(text))
                acc["entropy_4char"].append(char_ngram_entropy(text, 4))
                if decoder is not None and hasattr(decoder, "diagnostics"):
                    dg = decoder.diagnostics()
                    for k in diag:
                        diag[k].append(dg.get(k, float("nan")))

                samples_f.write(
                    f"[{c['name']}][{args.model}] prompt='{ptext}' seed={seed}\n"
                    + "-" * 60 + "\n" + text + "\n\n")

        row = {"strategy": c["name"], "n_samples": len(onsets)}
        for k, v in acc.items():
            m, lo, hi = cluster_bootstrap_ci(v, groups)
            row[f"{k}_mean"] = m; row[f"{k}_lo"] = lo; row[f"{k}_hi"] = hi
        for k, v in diag.items():
            if v and not all(x != x for x in v):
                m, lo, hi = cluster_bootstrap_ci(v, groups)
                row[f"{k}_mean"] = m; row[f"{k}_lo"] = lo; row[f"{k}_hi"] = hi
        row["loop_rate_persistent"] = loop_rate(onsets)
        row["survival_auc_persistent"] = survival_auc(onsets, args.n_tokens)
        row["rmst_persistent"] = rmst(onsets, args.n_tokens)
        row["onsets_persistent"] = onsets
        row["groups"] = groups
        if cps_val is not None:
            row["tokens_per_sec"] = cps_val
        rows.append(row)
        print(f"\r  {c['name']:<22} loop_rate={row['loop_rate_persistent']:.3f} "
              f"rmst={row['rmst_persistent']:.1f} "
              f"mcnll={row['mc_nll_bits_mean']:.3f}" + " " * 20)

    samples_f.close()

    skip = {"onsets_persistent", "groups"}
    keys = []
    for r in rows:
        for k in r:
            if k not in skip and k not in keys:
                keys.append(k)
    with open(os.path.join(args.out_dir, f"metrics_{args.model}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in rows])

    km = {}
    for r in rows:
        t, s = kaplan_meier(r["onsets_persistent"], args.n_tokens)
        km[r["strategy"]] = {"km_times": t, "km_survival": s,
                             "n_censored": sum(1 for x in r["onsets_persistent"] if x < 0),
                             "n_total": len(r["onsets_persistent"]),
                             "rmst": r["rmst_persistent"],
                             "event": f"persistent_cycle_R{R}_Pmax{period_max}_tokens"}
    with open(os.path.join(args.out_dir, "survival_curves.json"), "w") as f:
        json.dump(km, f, indent=2)

    with open(os.path.join(args.out_dir, "all_results.json"), "w") as f:
        json.dump({args.model: rows}, f, indent=2)

    print(f"\nWrote metrics_{args.model}.csv, survival_curves.json, "
          f"all_results.json, samples_{args.model}.txt -> {args.out_dir}/")


if __name__ == "__main__":
    main()
