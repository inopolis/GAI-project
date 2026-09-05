"""
modern_llm_comparison.py

Full decoder comparison on a MODERN, larger pretrained subword causal LM
(any HuggingFace model id, typically ~1-8B params -- e.g. an OLMo, Qwen, or
Mistral checkpoint), run only AFTER pretrained_subword_pilot.py has
confirmed a measurable low-temperature repetition regime and
validate_loop_event_subword.py has frozen THIS model's OWN (R, period_max).
Reusing gpt2_full_comparison.py's frozen event for a different model family
would repeat exactly the mistake already found and fixed once in this
project (assuming a threshold transfers instead of checking) -- hence
--event_report below has no GPT-2-specific default; the caller must point
it at the report produced for the model actually being run here.

Relationship to the two existing scripts this one merges
----------------------------------------------------------
  - gpt2_full_comparison.py supplies the STRUCTURE: the decoder-config list
    at the token-level operating point (temperature=0.10 held for every
    non-sweep config, n_min=2/n_max=4 for token-level risk, the same
    hyperparameter values), the per-prompt x per-seed loop, loop-onset /
    RMST / survival-curve bookkeeping via sampling_eval.py, and the
    metrics-CSV / all_results.json / samples-file save format.
  - common_kl_comparison.py supplies the SCORING METHODOLOGY: a shared
    reference distribution p (temperature-scaled softmax of the RAW,
    decoder-untouched model logits, T_REF=0.10 fixed regardless of what
    temperature any individual decoder itself uses), the decoder's actual
    sampling distribution q read from decoder.last_q (never re-derived),
    KL(q||p) via the NaN-safe 0*log(0):=0 convention (safe_kl_bits, copied
    verbatim below), a PASSIVE shadow RecurrenceRiskDecoder that is never
    .step()'d and exists only to track the one standard recurrence-risk
    definition independently of whichever real decoder is driving
    generation, and per-step latency timed around exactly decoder.step().
    Baseline sampling strategies (plain temperature, repetition penalty,
    no-repeat-ngram, greedy) are wrapped as thin classes exposing the same
    reset/prime/step/last_q interface as the class-based decoders, exactly
    as common_kl_comparison.py does, so every config -- baseline or
    decoder-class -- is driven and scored through one identical code path.
    These wrapper classes and safe_kl_bits are duplicated (not imported)
    from common_kl_comparison.py -- same discipline gpt2_full_comparison.py
    itself uses for persistent_loop_onset_tokens ("duplicated ... because
    that module is a standalone calibration script, not a library; kept
    byte-identical on purpose"): importing a sibling top-level script here
    would work today, but ties this script's correctness to that other
    script never restructuring its module-level code, for no real benefit.

  Shadow risk order: common_kl_comparison.py's shadow uses n_min=3/n_max=6,
  the CHARACTER-level convention (a repeated span of 3-6 characters is a
  meaningfully short unit). This script instead uses n_min=2/n_max=4,
  matching every token-level risk config in gpt2_full_comparison.py's own
  build_configs() -- a token already carries several characters' worth of
  information, so the token-level "standard risk" order is shorter than the
  character-level one, not a literal copy of it. SHADOW_N_MIN/SHADOW_N_MAX
  below make this an explicit, named choice rather than a silent constant.

The ONE genuinely new piece of engineering here, not present in either
parent script: KV-CACHED GENERATION.
--------------------------------------------------------------------------
gpt2_full_comparison.py's generate_hf() recomputes logits for the ENTIRE
growing sequence every step (`model(x[:, -model.block_size:])`, no
past_key_values) -- O(n) work per step, O(n^2) total, harmless for GPT-2's
124M params over 200 tokens but far too slow to run many prompts x seeds x
configs against a 1-8B+ parameter model. generate_hf_kv() below instead:
  1. Primes the cache ONCE per sample with a single forward pass over the
     full prompt: `model(input_ids=<prompt>, use_cache=True)`, keeping
     `outputs.past_key_values` (a `Cache`/`DynamicCache` object under
     current `transformers` -- treated here as an opaque token passed back
     in, never indexed or assumed to be a plain tuple of tensors, so this
     does not depend on the older tuple-of-tuples KV-cache API).
  2. On every subsequent step, feeds ONLY the single most recently chosen
     token: `model(input_ids=<that one token>, past_key_values=cache,
     use_cache=True)`, and carries the RETURNED `outputs.past_key_values`
     forward into the next step -- O(1) new work per step (attention over
     the cache is internal to the forward pass and is the whole point of
     the cache), O(n) total.
  3. Every step still reads `outputs.logits[0, -1, :]`, exactly like the
     two parent scripts -- the (1, seq_len, vocab) -> last-position-logits
     convention is unchanged; only how those logits get computed changes.

mcNLL is folded into the SAME pass instead of gpt2_full_comparison.py's
separate post-hoc rescoring forward pass. That rescoring pass is
mathematically redundant here: gpt2_full_comparison.py's mc_nll_bits(model,
full_ids, prompt_len, device) teacher-forces the WHOLE prompt+generation
through the model a second time and reads off -log2 p(token_t | token_<t)
for t >= prompt_len. But generation itself, autoregressively conditioned on
its own previously emitted tokens via the cache, computes EXACTLY that same
conditional -- the RAW (temperature-1, decoder-untouched) model logits at
each generation step already give p(token_t | token_<t) for the token
actually emitted next. So mc_nll_bits' definition is reused verbatim (bits
per generated token, under the model's own unmodified distribution) but its
VALUE is accumulated online, one term per step, from logits this script has
already computed for the KL/risk scoring above -- adapting it to avoid a
second O(n) forward pass per sample would otherwise cost, multiplied across
every prompt x seed x config, exactly the kind of expense KV-caching this
script exists to avoid in the first place.

Usage
-----
  python3 modern_llm_comparison.py --model allenai/OLMo-2-1124-7B \
      --event_report loop_event_subword_report_olmo/loop_event_subword_report.json \
      --n_seeds 10 --n_tokens 200 --out_dir runs/modern_llm_comparison_olmo
"""
import os, sys, argparse, csv, json, math, time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.decoding import (SuffixMatchDecoder, FSDDecoder, LZPenaltyDecoder,
                          RecurrenceRiskDecoder, repetition_penalty_filtering,
                          no_repeat_ngram_filtering)
# Reused verbatim -- same rationale as gpt2_full_comparison.py's module
# docstring: these are generic (model-agnostic) and duplicating them here
# would risk the comparisons silently diverging in definition.
from sampling_eval import (cluster_bootstrap_ci, kaplan_meier, survival_auc,
                           rmst, loop_rate, compression_ratio,
                           longest_repeated_substring, char_ngram_entropy,
                           spelling_error_rate)


# ---------------------------------------------------------------------------
# PROMPT PROTOCOL. Fresh prompts, written after the fact and never inspected
# while choosing any hyperparameter above -- and deliberately NOT reused from
# gpt2_full_comparison.py's TEST_PROMPTS or pretrained_subword_pilot.py's
# PILOT_PROMPTS, to avoid the tuning/reporting leakage this project's other
# scripts are careful to avoid (see gpt2_full_comparison.py's own comment on
# this same discipline).
# ---------------------------------------------------------------------------
TEST_PROMPTS = [
    ("m_committee",  "The committee announced that"),
    ("m_scientists", "Scientists recently discovered that"),
    ("m_research",   "After years of research, the team"),
    ("m_novel",      "The novel begins when a stranger"),
    ("m_investig",   "Investigators later determined that"),
    ("m_spokesperson","The company's spokesperson stated that"),
    ("m_beneath",    "Deep beneath the surface, researchers found"),
    ("m_negotiate",  "The negotiations concluded when both sides"),
    ("m_historians", "Historians have long debated whether"),
    ("m_festival",   "The festival opened with a speech about"),
]

T_REF = 0.10  # fixed shared reference temperature for p, matching
              # common_kl_comparison.py's / gpt2_full_comparison.py's shared
              # operating point -- NOT each decoder's own temperature.
SHADOW_N_MIN, SHADOW_N_MAX = 2, 4  # standard risk order for the passive
              # shadow tracker -- see module docstring for why this is the
              # token-level (not character-level) convention.


def load_frozen_event(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"Missing {path}. Run validate_loop_event_subword.py for THIS "
            f"model first -- this script refuses to fall back to a "
            f"placeholder (R, period_max), or to silently reuse another "
            f"model's frozen event, rather than an event calibrated for the "
            f"model actually being run here.")
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
    """Identical logic to validate_loop_event_subword.py's / gpt2_full_
    comparison.py's frozen event -- duplicated (not imported), same
    rationale as gpt2_full_comparison.py's copy of this function: these are
    standalone scripts, not a shared library, and this is kept
    byte-identical on purpose."""
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


def safe_kl_bits(q, log_p):
    """Same 0*log(0):=0 convention as RecurrenceRiskDecoder._kl_bits /
    common_kl_comparison.py's safe_kl_bits (copied verbatim, same rationale:
    taking q directly, a probability tensor every decoder's/baseline's
    last_q already is, rather than log_q, since not every decoder here
    computes q via log-space)."""
    log_q = torch.log(torch.clamp(q, min=1e-45))
    term = q * (log_q - log_p)
    return float(torch.where(q > 0, term, torch.zeros_like(term)).sum() / math.log(2.0))


# ---------------------------------------------------------------------------
# Thin baseline wrappers exposing the SAME reset/prime/step/last_q interface
# as the class-based decoders (SuffixMatchDecoder/FSDDecoder/LZPenaltyDecoder/
# RecurrenceRiskDecoder), duplicated from common_kl_comparison.py (see module
# docstring for why duplicated, not imported) so every config in this
# script -- baseline or decoder-class -- is driven and scored through one
# identical KV-cached generation loop.
# ---------------------------------------------------------------------------

class GreedyBaseline:
    """Deterministic argmax. last_q is the (proper, one-hot) degenerate
    distribution actually sampled from -- not a smoothed approximation --
    so KL(q||p) and E_q[risk] remain well-defined via the same safe_kl_bits
    formula as every other decoder here (q>0 only at the argmax token, and
    softmax over real logits never produces an exact p==0 there)."""
    def reset(self): pass
    def prime(self, ids): pass
    def step(self, logits, generated_ids=None):
        tok = int(torch.argmax(logits).item())
        q = torch.zeros_like(logits)
        q[tok] = 1.0
        self.last_q = q
        return tok


class TemperatureBaseline:
    """Plain temperature sampling. When temperature == T_REF this is the
    T_REF reference itself: q == p exactly, KL == 0 exactly by
    construction -- used below as the temperature_reference sanity check on
    the whole KL/risk harness (any nonzero KL there indicates a bug in how
    p/q are computed elsewhere), matching common_kl_comparison.py's
    TemperatureBaseline docstring/rationale exactly."""
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
    """Hard constraint -- see src/decoding.py's docstring: an all-banned
    vocabulary at this n would raise on torch.multinomial. Not expected in
    practice at n=3 on a modern, large vocabulary, same accepted risk as
    common_kl_comparison.py's NoRepeatBaseline."""
    def __init__(self, temperature=0.10, n=3):
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


def make_configs(vocab_size):
    """Mirrors gpt2_full_comparison.py's build_configs() operating point
    (temperature=0.10 held for every non-sweep config, n_min=2/n_max=4 for
    token-level risk, identical alpha/eps/lambda values), but returns a dict
    of zero-arg factories (name -> callable returning a fresh decoder/
    baseline instance) so every config shares one driving/scoring loop.
    LZPenaltyDecoder needs vocab_size at construction time (see
    src/decoding.py), hence this function takes it as a parameter rather
    than being a module-level constant, matching gpt2_full_comparison.py's
    make_decoder(cfg, vocab_size)."""
    C = {
        "greedy": lambda: GreedyBaseline(),
        "temperature_reference": lambda: TemperatureBaseline(temperature=T_REF),
    }
    for t in (0.05, 0.10, 0.15):
        C[f"lt_temp_{t}"] = (lambda t=t: TemperatureBaseline(temperature=t))
    C["lt_rep_penalty_1.3"] = lambda: RepPenaltyBaseline(temperature=0.10, penalty=1.3)
    C["lt_no_repeat_3gram"] = lambda: NoRepeatBaseline(temperature=0.10, n=3)
    C["lt_suffixmatch_a3.0"] = lambda: SuffixMatchDecoder(
        temperature=0.10, top_p=1.0, alpha=3.0, max_history=400, ref_len=20)
    C["lt_fsd"] = lambda: FSDDecoder(
        temperature=0.10, top_p=1.0, alpha=4.0, n_min=2, n_max=4)
    C["lt_lzpenalty"] = lambda: LZPenaltyDecoder(
        temperature=0.10, top_p=1.0, alpha=0.15,
        buffer_size=32, window_size=512, vocab_size=vocab_size)
    C["lt_risk_only"] = lambda: RecurrenceRiskDecoder(
        temperature=0.10, top_p=1.0, n_min=2, n_max=4,
        mode="fixed", alpha_base=3.0, include_prompt_context=True)
    C["lt_adaptive"] = lambda: RecurrenceRiskDecoder(
        temperature=0.10, top_p=1.0, n_min=2, n_max=4,
        mode="adaptive", alpha_base=3.0, alpha_max=12.0,
        lambda_rep=15.0, lambda_ent=1.0, rep_target=0.05,
        ent_target=4.0, window=64)
    for e in (0.01, 0.05):
        C[f"lt_dual_eps{e}"] = (lambda e=e: RecurrenceRiskDecoder(
            temperature=0.10, top_p=1.0, n_min=2, n_max=4, mode="dual", eps=e))
    return C


@torch.no_grad()
def generate_and_score_kv(model, prompt_ids, n_tokens, device, decoder, shadow):
    """
    KV-cached generation loop with the common-distortion instrumentation
    folded in, one forward pass per NEW token (not per growing sequence --
    see module docstring). `shadow` is a PASSIVE RecurrenceRiskDecoder,
    never .step()'d, used purely for its ._risk_scores()/._register()
    bookkeeping so every config's risk is measured against the SAME
    standard risk definition regardless of what that config's own mechanism
    (if any) internally does with risk -- identical role to
    common_kl_comparison.py's shadow.

    Returns (full_ids, kl_bits, risk_achieved, latency_ms, nll_bits):
    per-step lists, one entry per generated token.
    """
    ids = list(prompt_ids)
    decoder.reset()
    # SuffixMatchDecoder has no prime() -- it reads history straight from
    # the generated_ids argument passed to step() every call, so there is
    # nothing to pre-register. Matches src/decoding.py's generate() and
    # both parent scripts, which likewise never call .prime() on it.
    if hasattr(decoder, "prime"):
        decoder.prime(ids)
    shadow.reset(); shadow.prime(ids)

    x = torch.tensor([ids], device=device)
    out = model(input_ids=x, use_cache=True)
    logits = out.logits[0, -1, :]
    past = out.past_key_values  # opaque Cache object -- passed back in
                                 # verbatim, never indexed as a tuple.

    kl_bits, risk_achieved, latency_ms, nll_bits = [], [], [], []
    for _ in range(n_tokens):
        log_p_ref = F.log_softmax(logits / max(T_REF, 1e-6), dim=-1)
        risk_vec = shadow._risk_scores(logits.shape[-1]).to(logits.device)
        # RAW (T=1, decoder-untouched) model distribution -- for mcNLL, the
        # self-consistency NLL of the token about to be emitted under the
        # model's own unmodified next-token distribution at this context.
        raw_log_p = F.log_softmax(logits, dim=-1)

        t0 = time.perf_counter()
        tok = decoder.step(logits, ids)
        latency_ms.append((time.perf_counter() - t0) * 1000.0)

        q = decoder.last_q
        kl_bits.append(safe_kl_bits(q, log_p_ref))
        risk_achieved.append(float((q * risk_vec).sum()))
        nll_bits.append(float(-raw_log_p[tok].item() / math.log(2.0)))

        ids.append(tok)
        shadow._register(tok)

        next_in = torch.tensor([[tok]], device=device)
        out = model(input_ids=next_in, past_key_values=past, use_cache=True)
        logits = out.logits[0, -1, :]
        past = out.past_key_values

    return ids, kl_bits, risk_achieved, latency_ms, nll_bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HuggingFace model id, no default -- this script "
                         "runs across multiple model families and a "
                         "GPT-2 default here would be actively misleading.")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--n_tokens", type=int, default=200)
    ap.add_argument("--out_dir", default="runs/modern_llm_comparison")
    ap.add_argument("--event_report", required=False, default=None,
                    help="Path to THIS model's loop_event_subword_report.json "
                         "(from validate_loop_event_subword.py --out_dir ...). "
                         "No GPT-2-specific default -- see module docstring.")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None,
                    help="Override the default dtype (bfloat16 on CUDA, "
                         "float32 everywhere else, deliberately -- see the "
                         "comment above the dtype selection in main() for "
                         "why float32 is the default on MPS too, not just "
                         "CPU). Not set by default; pass explicitly to trade "
                         "precision consistency for memory/speed.")
    args = ap.parse_args()

    if args.event_report is None:
        raise SystemExit(
            "--event_report is required: point it at the "
            "loop_event_subword_report.json produced by "
            "validate_loop_event_subword.py for THIS model (--model "
            f"{args.model}). There is no GPT-2-specific default here.")

    R, period_max = load_frozen_event(args.event_report)
    print(f"Frozen token-level event for {args.model}: R={R}, period_max={period_max}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps") if torch.backends.mps.is_available() else
              torch.device("cpu"))
    # float32 everywhere by default, deliberately, even though float16 was
    # verified safe on MPS (safe_kl_bits' 0*log(0):=0 guard via
    # torch.where stays NaN-free even when its clamp floor underflows to
    # exactly 0.0 in fp16 -- confirmed directly). Kept at float32 anyway:
    # (1) this project's central finding IS a float32-underflow bug in the
    # dual solver, found on review -- deliberately lowering precision
    # further for a different experiment in the same paper invites exactly
    # the kind of scrutiny that finding earns, and (2) running the three
    # modern-LLM families at different precisions (OLMo already committed
    # to float32) would confound model family with numerical precision,
    # undermining the cross-family comparison this experiment exists to
    # make. bfloat16 remains the CUDA default (unaffected by either
    # concern: no MPS involved, and this project has never run a CUDA
    # experiment to be inconsistent with). --dtype below is an explicit,
    # opt-in override, not a default, for a future run to change if the
    # tradeoff is judged worth it.
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if args.dtype is not None:
        dtype = dtype_map[args.dtype]
    elif device.type == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    print(f"Loading {args.model} on {device} (dtype={dtype})...")
    # device_map=... loads each weight shard straight onto the target device
    # instead of materializing the full model on CPU first and then copying
    # it with .to(device) -- the old pattern peaks at ~2x model size in
    # memory (full CPU copy + full target-device copy coexisting mid-copy),
    # which is what pushed Mistral-7B into swap on 36GB unified memory.
    # Same dtype, same target device, same numerics -- only the loading
    # path changes.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True, device_map={"": device},
    ).eval()
    vocab_size = model.config.vocab_size

    os.makedirs(args.out_dir, exist_ok=True)
    model_basename = args.model.split("/")[-1]
    configs = make_configs(vocab_size)
    if args.only:
        configs = {k: v for k, v in configs.items() if k in set(args.only)}

    samples_f = open(os.path.join(args.out_dir, f"samples_{model_basename}.txt"),
                     "w", encoding="utf-8")
    rows = []
    for name, factory in configs.items():
        print(f"  {name:<24}", end="", flush=True)
        onsets, groups = [], []
        acc = {"rep_mass_3": [], "compression": [], "longest_rep_sub": [],
               "spelling_error": [], "entropy_4char": [], "mc_nll_bits": [],
               "kl_bits": [], "risk_achieved": [], "latency_ms": []}
        # RecurrenceRiskDecoder-internal diagnostics not otherwise available
        # from the common-distortion columns above (dual solver's own
        # operating point) -- decoder-internal kl_bits/risk_achieved are
        # deliberately NOT pulled in here: they are computed against the
        # decoder's OWN self.temperature, a different reference than the
        # shared T_REF-based acc["kl_bits"]/acc["risk_achieved"] above, and
        # mixing the two under the same column name would be misleading.
        diag = {"lambda_mean": [], "dual_feasible_rate": [],
                "dual_structurally_infeasible_rate": [],
                "dual_near_boundary_rate": []}
        tps_val = None
        gen_ids_all = []

        for pi, (pname, ptext) in enumerate(TEST_PROMPTS):
            for seed in range(1, args.n_seeds + 1):
                print(f"\r  {name:<24} sample {pi*args.n_seeds+seed}/"
                      f"{len(TEST_PROMPTS)*args.n_seeds}", end="", flush=True)
                torch.manual_seed(seed * 1000 + pi)
                np.random.seed(seed * 1000 + pi)
                prompt_ids = tok(ptext, return_tensors="pt").input_ids[0].tolist()
                decoder = factory()
                shadow = RecurrenceRiskDecoder(n_min=SHADOW_N_MIN, n_max=SHADOW_N_MAX,
                                               mode="fixed", alpha_base=0.0)

                measure = (pi == 0 and seed == 1)
                t0 = time.perf_counter() if measure else None
                full_ids, kls, risks, lats, nlls = generate_and_score_kv(
                    model, prompt_ids, args.n_tokens, device, decoder, shadow)
                if measure:
                    elapsed = time.perf_counter() - t0
                    tps_val = round(args.n_tokens / elapsed, 2) if elapsed > 0 else 0.0

                gen_ids = full_ids[len(prompt_ids):]
                text = tok.decode(gen_ids)

                groups.append(pi)
                gen_ids_all.append(gen_ids)
                onsets.append(persistent_loop_onset_tokens(gen_ids, R, period_max))
                acc["rep_mass_3"].append(rep_ngram_mass_tokens(gen_ids, 3))
                acc["compression"].append(compression_ratio(text))
                acc["longest_rep_sub"].append(longest_repeated_substring(text))
                acc["spelling_error"].append(spelling_error_rate(text))
                acc["entropy_4char"].append(char_ngram_entropy(text, 4))
                acc["mc_nll_bits"].append(float(np.mean(nlls)) if nlls else float("nan"))
                acc["kl_bits"].append(float(np.mean(kls)) if kls else float("nan"))
                acc["risk_achieved"].append(float(np.mean(risks)) if risks else float("nan"))
                acc["latency_ms"].append(float(np.mean(lats)) if lats else float("nan"))
                if hasattr(decoder, "diagnostics"):
                    dg = decoder.diagnostics()
                    for k in diag:
                        diag[k].append(dg.get(k, float("nan")))

                samples_f.write(
                    f"[{name}][{args.model}] prompt='{ptext}' seed={seed}\n"
                    + "-" * 60 + "\n" + text + "\n\n")

        row = {"strategy": name, "n_samples": len(onsets)}
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
        # Raw generated token ids, kept for post-hoc detector-threshold
        # (R, period_max) sensitivity sweeps without re-generating -- decoded
        # text alone is not reliably re-tokenizable (empirically checked:
        # re-encoding saved sample text reproduced a different token count
        # than the original generation for 18% of Mistral samples and 2% of
        # OLMo samples, so text-only storage cannot support this later).
        row["gen_ids"] = gen_ids_all
        if tps_val is not None:
            row["tokens_per_sec"] = tps_val
        rows.append(row)
        print(f"\r  {name:<24} loop_rate={row['loop_rate_persistent']:.3f} "
              f"rmst={row['rmst_persistent']:.1f} "
              f"kl={row['kl_bits_mean']:.4f}b "
              f"mcnll={row['mc_nll_bits_mean']:.3f}" + " " * 15)

    samples_f.close()

    skip = {"onsets_persistent", "groups", "gen_ids"}
    keys = []
    for r in rows:
        for k in r:
            if k not in skip and k not in keys:
                keys.append(k)
    with open(os.path.join(args.out_dir, f"metrics_{model_basename}.csv"), "w", newline="") as f:
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

    print(f"\nWrote metrics_{model_basename}.csv, survival_curves.json, "
          f"all_results.json, samples_{model_basename}.txt -> {args.out_dir}/")


if __name__ == "__main__":
    main()
