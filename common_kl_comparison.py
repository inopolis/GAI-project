"""
common_kl_comparison.py

The fair common-distortion comparison requested on review: compute the
FINAL next-token KL(q||p), the standard recurrence risk E_q[risk], and
per-step latency for every decoder -- LZ Penalty, FSD-style, repetition
penalty, SuffixMatch, and all recurrence-risk (RR) modes -- against the
SAME reference distribution p, not just for the RR family.

Why this did not already exist: only RecurrenceRiskDecoder computed and
reported KL(q||p) at all (sampling_eval.py's tables show "---" for every
other decoder's KL column). LZ Penalty, FSD, and SuffixMatch all sample
from a well-defined q at every step (temperature + top-p over an adjusted
logit vector) -- that q was simply never compared to p or scored against
the risk definition used elsewhere in this project. Repetition penalty and
plain temperature sampling likewise define a q with no KL/risk figure
attached anywhere.

Design
------
For every decoder, at every step:
  1. Get the SAME raw model logits every other decoder would see at this
     context (decoders never share state across configs -- each config
     regenerates from the same checkpoint, prompts, and seeds).
  2. p := temperature-scaled reference distribution, softmax(logits / T),
     T = 0.10 throughout, matching Table~\\ref{tab:main}'s operating point.
     This is NOT any decoder's output -- it is the common reference every
     decoder's q is measured against, exactly as the dual theorem's (P)
     measures distortion from p, not from any other decoder.
  3. q := the decoder's ACTUAL sampling distribution this step, read from
     `decoder.last_q` (added to every decoder class specifically for this
     comparison -- see src/decoding.py) -- not re-derived or approximated.
  4. risk := the STANDARD multi-order recurrence risk vector (same
     n_min/n_max, same incremental hash-map definition as
     RecurrenceRiskDecoder._risk_scores), computed by a PASSIVE shadow
     tracker that observes every decoder's actual generation history but
     never influences it -- so E_q[risk] is comparable across every
     decoder using one shared risk definition, not each decoder's own
     (possibly decoder-specific) notion of risk.
  5. KL(q||p) and E_q[risk] computed via the SAME safe (0*log(0):=0)
     formula added to RecurrenceRiskDecoder._kl_bits during this review
     pass, reused here rather than re-derived, so both places are
     guaranteed consistent.
  6. Per-step latency: wall-clock time of exactly the decoder.step() call,
     isolated from model forward-pass time (timed once, shared across all
     decoders at that step) and from metric bookkeeping.

Usage
-----
  python3 common_kl_comparison.py --ckpt runs/cosine/best.pt \
      --data_dir data_out --out_dir runs/common_kl_comparison \
      --n_seeds 10 --n_chars 500 --prompt_set test
"""
import os, sys, argparse, csv, json, math, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from src.utils import set_seed, load_json, ensure_dir
from src.model import CharTransformerLM
from src.decoding import (RecurrenceRiskDecoder, SuffixMatchDecoder, FSDDecoder,
                          LZPenaltyDecoder, top_p_filtering,
                          repetition_penalty_filtering, no_repeat_ngram_filtering)
from sampling_eval import (TEST_PROMPTS, DEV_PROMPTS, encode, decode, load_model,
                           cluster_bootstrap_ci, persistent_loop_onset,
                           survival_auc, rmst, loop_rate)


# ---------------------------------------------------------------------------
# Thin baseline wrappers exposing the SAME reset/prime/step/last_q interface
# as the class-based decoders, so the harness below drives all eight configs
# identically. Each reuses the EXISTING filtering function from
# src/decoding.py rather than reimplementing the baseline's logic --
# guarantees this comparison scores the SAME repetition-penalty/no-repeat
# behaviour reported elsewhere in this project, not a re-derived variant.
# ---------------------------------------------------------------------------

class TemperatureBaseline:
    """Plain temperature sampling -- the T=0.10 reference itself as a
    decoder: q == p exactly, KL == 0 exactly, by construction. Included as
    a sanity check on the harness (any nonzero KL here would indicate a bug
    in how p/q are being computed elsewhere)."""
    def __init__(self, temperature=0.10):
        self.temperature = temperature
    def reset(self): pass
    def prime(self, ids): pass
    def step(self, logits, generated_ids=None):
        probs = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        self.last_q = probs
        return int(torch.multinomial(probs, 1).item())


class RepPenaltyBaseline:
    def __init__(self, temperature=0.10, penalty=1.3):
        self.temperature = temperature
        self.penalty = penalty
    def reset(self):
        self._ids = []
    def prime(self, ids):
        self._ids = list(ids)
    def step(self, logits, generated_ids=None):
        adjusted = repetition_penalty_filtering(logits.clone(), self._ids, self.penalty)
        probs = torch.softmax(adjusted / max(self.temperature, 1e-6), dim=-1)
        self.last_q = probs
        token = int(torch.multinomial(probs, 1).item())
        self._ids.append(token)
        return token


class NoRepeatBaseline:
    """Hard constraint -- KL(q||p) can be exactly log(0)-undefined-turned-inf
    if EVERY token is banned (Section 5's boundary discussion applies here
    too); in practice, at n=4 on this vocabulary, at most a handful of
    tokens are ever banned, so q stays a proper distribution over survivors."""
    def __init__(self, temperature=0.10, n=4):
        self.temperature = temperature
        self.n = n
    def reset(self):
        self._ids = []
    def prime(self, ids):
        self._ids = list(ids)
    def step(self, logits, generated_ids=None):
        adjusted = no_repeat_ngram_filtering(logits.clone(), self._ids, self.n)
        probs = torch.softmax(adjusted / max(self.temperature, 1e-6), dim=-1)
        self.last_q = probs
        token = int(torch.multinomial(probs, 1).item())
        self._ids.append(token)
        return token


def safe_kl_bits(q, log_p):
    """Same 0*log(0):=0 convention as RecurrenceRiskDecoder._kl_bits
    (Section~\\ref{ssec:dual-fix}'s NaN fix), taking q directly (a
    probability tensor, as every decoder's last_q already is) rather than
    log_q, since not every decoder here computes q via log-space."""
    log_q = torch.log(torch.clamp(q, min=1e-45))
    term = q * (log_q - log_p)
    return float(torch.where(q > 0, term, torch.zeros_like(term)).sum() / math.log(2.0))


def make_configs():
    """LZ, FSD, repetition penalty, SuffixMatch, and all RR modes, per
    review -- plus the T=0.10 reference itself as a sanity check. All at
    T=0.10, matching Table~\\ref{tab:main}'s operating point, so this is a
    like-for-like common-distortion comparison at the SAME temperature the
    headline loop-rate numbers were measured at, not a different regime."""
    return {
        "temperature_0.10": lambda: TemperatureBaseline(temperature=0.10),
        "rep_penalty_1.3":  lambda: RepPenaltyBaseline(temperature=0.10, penalty=1.3),
        "no_repeat_4gram":  lambda: NoRepeatBaseline(temperature=0.10, n=4),
        "suffixmatch_a3.0": lambda: SuffixMatchDecoder(temperature=0.10, top_p=1.0, alpha=3.0,
                                                        max_history=400, ref_len=20),
        "fsd":              lambda: FSDDecoder(temperature=0.10, top_p=1.0, alpha=4.0,
                                               n_min=3, n_max=6),
        "lzpenalty":        lambda: LZPenaltyDecoder(temperature=0.10, top_p=1.0, alpha=0.15,
                                                      buffer_size=8, window_size=128),
        "rr_fixed":         lambda: RecurrenceRiskDecoder(temperature=0.10, top_p=1.0, n_min=3, n_max=6,
                                                           mode="fixed", alpha_base=2.0),
        "rr_adaptive":      lambda: RecurrenceRiskDecoder(temperature=0.10, top_p=1.0, n_min=3, n_max=6,
                                                           mode="adaptive", alpha_base=2.0, alpha_max=8.0,
                                                           lambda_rep=10.0, lambda_ent=1.0,
                                                           rep_target=0.05, ent_target=3.5, window=100),
        "rr_dual_eps0.05":  lambda: RecurrenceRiskDecoder(temperature=0.10, top_p=1.0, n_min=3, n_max=6,
                                                           mode="dual", eps=0.05),
        "rr_dual_eps0.01":  lambda: RecurrenceRiskDecoder(temperature=0.10, top_p=1.0, n_min=3, n_max=6,
                                                           mode="dual", eps=0.01),
    }


@torch.no_grad()
def generate_and_score(model, prompt_ids, n_chars, device, decoder, shadow,
                       temperature=0.10):
    """Generation loop with the common-KL instrumentation. `shadow` is a
    PASSIVE RecurrenceRiskDecoder used purely for its ._risk_scores()/
    ._register() bookkeeping (mode is irrelevant -- .step() is never
    called on it), so every config's risk is measured against the SAME
    standard risk definition regardless of what that config's own
    mechanism (if any) internally does with risk."""
    ids = list(prompt_ids)
    x = torch.tensor([ids], device=device)
    decoder.reset()
    # SuffixMatchDecoder has no prime() -- unlike the other decoders here, it
    # reads history straight from the generated_ids argument passed to
    # step() every call, so there is nothing to pre-register. Matches
    # src/decoding.py's generate(), which never calls .prime() on its
    # lz_decoder slot either.
    if hasattr(decoder, "prime"):
        decoder.prime(ids)
    shadow.reset(); shadow.prime(ids)

    kl_bits, risk_achieved, latency_ms = [], [], []
    for _ in range(n_chars):
        logits = model(x[:, -model.block_size:])[0][0, -1, :]
        log_p = F.log_softmax(logits / max(temperature, 1e-6), dim=-1)
        risk_vec = shadow._risk_scores(logits.shape[-1]).to(logits.device)

        t0 = time.perf_counter()
        tok = decoder.step(logits, ids)
        latency_ms.append((time.perf_counter() - t0) * 1000.0)

        q = decoder.last_q
        kl_bits.append(safe_kl_bits(q, log_p))
        risk_achieved.append(float((q * risk_vec).sum()))

        ids.append(tok)
        shadow._register(tok)
        x = torch.cat([x, torch.tensor([[tok]], device=device)], dim=1)

    return ids, kl_bits, risk_achieved, latency_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/cosine/best.pt")
    ap.add_argument("--data_dir", default="data_out")
    ap.add_argument("--out_dir", default="runs/common_kl_comparison")
    ap.add_argument("--n_chars", type=int, default=500)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--prompt_set", choices=["dev", "test"], default="test")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    prompts = TEST_PROMPTS if args.prompt_set == "test" else DEV_PROMPTS
    device = (torch.device("mps") if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}  Prompts: {len(prompts)}  Seeds: {args.n_seeds}")

    model, cfg = load_model(args.ckpt, device)
    vocab = load_json(os.path.join(args.data_dir, "vocab.json"))
    stoi, itos = vocab["stoi"], vocab["itos"]
    ckpt_name = os.path.basename(os.path.dirname(args.ckpt))

    ensure_dir(args.out_dir)
    configs = make_configs()
    if args.only:
        configs = {k: v for k, v in configs.items() if k in set(args.only)}

    rows = []
    for name, factory in configs.items():
        print(f"  {name:<20}", end="", flush=True)
        kl_all, risk_all, lat_all, onsets, groups = [], [], [], [], []
        for pi, (pname, ptext) in enumerate(prompts):
            for seed in range(1, args.n_seeds + 1):
                print(f"\r  {name:<22} sample {pi*args.n_seeds+seed}/"
                      f"{len(prompts)*args.n_seeds}", end="", flush=True)
                set_seed(seed)
                prompt_ids = encode(ptext, stoi)[0].tolist()
                decoder = factory()
                shadow = RecurrenceRiskDecoder(n_min=3, n_max=6, mode="fixed", alpha_base=0.0)
                gen_ids, kls, risks, lats = generate_and_score(
                    model, prompt_ids, args.n_chars, device, decoder, shadow)
                text = decode(gen_ids[len(prompt_ids):], itos)

                kl_all.append(float(np.mean(kls)))
                risk_all.append(float(np.mean(risks)))
                lat_all.append(float(np.mean(lats)))
                onsets.append(persistent_loop_onset(text))
                groups.append(pi)

        row = {"strategy": name, "checkpoint": ckpt_name, "n_samples": len(kl_all)}
        for label, vals in [("kl_bits", kl_all), ("risk_achieved", risk_all),
                            ("latency_ms", lat_all)]:
            m, lo, hi = cluster_bootstrap_ci(vals, groups)
            row[f"{label}_mean"] = m; row[f"{label}_lo"] = lo; row[f"{label}_hi"] = hi
        row["loop_rate"] = loop_rate(onsets)
        row["rmst"] = rmst(onsets, args.n_chars)
        row["survival_auc"] = survival_auc(onsets, args.n_chars)
        rows.append(row)
        print(f"\r  {name:<22} KL={row['kl_bits_mean']:.4f}b  "
              f"risk={row['risk_achieved_mean']:.4f}  "
              f"latency={row['latency_ms_mean']:.3f}ms  "
              f"loop={row['loop_rate']:.3f}" + " " * 15)

    keys = list(rows[0].keys()) if rows else []
    with open(os.path.join(args.out_dir, "common_kl_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out_dir, "common_kl_comparison.json"), "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\nWrote common_kl_comparison.{{csv,json}} -> {args.out_dir}/")
    print("\nPareto view (loop_rate vs KL, sorted by KL):")
    for r in sorted(rows, key=lambda r: r["kl_bits_mean"]):
        print(f"  {r['strategy']:<20} KL={r['kl_bits_mean']:>8.4f}b  "
              f"risk={r['risk_achieved_mean']:>7.4f}  loop={r['loop_rate']:>6.3f}  "
              f"latency={r['latency_ms_mean']:>7.3f}ms")


if __name__ == "__main__":
    main()
