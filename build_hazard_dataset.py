"""
build_hazard_dataset.py

Builds an offline, rollout-labeled dataset for training a finite-horizon
loop-hazard estimator, replacing the infeasible per-step Monte Carlo rollout
(which would cost ~150,000 forward passes per generated sample) with a
one-time, batched, offline data-collection pass.

Design
------
1. CONTEXTS: sampled from already-generated text (reusing samples_cosine.txt
   files from prior runs, or fresh raw-temperature generations if none are
   available), at varied positions -- some near a real loop onset, some far
   from any looping, so the label distribution isn't degenerate.
2. ROLLOUT: from each context, continue generating for H steps under the
   RAW reference distribution (temperature=0.10, no intervention -- this
   must match the reference distribution p that the dual/fixed/adaptive
   projection operates on) and check with the FROZEN, whitespace-normalized
   persistent_loop_onset event whether a loop occurs within those H steps.
   K independent rollouts per context give a Monte Carlo estimate of the
   true hazard label (majority vote, or the raw K/n rate for soft labels).
3. FEATURES: cheap, already-computed-elsewhere signals, so hazard inference
   at decoding time costs one forward pass through a tiny model, not a
   rollout: recent repetition rate, recent entropy, current exact-risk
   value, longest-suffix look-back match length. All are O(window) to
   compute from the trailing context, no model forward pass needed beyond
   what decoding already does.
4. BATCHING: all N*K rollouts run as one batched generation (batch dimension
   = N*K, sequence dimension = H), so wall-clock is ~H sequential steps
   total, not N*K*H.

Output: a JSON/CSV file of (features, label) pairs for train_hazard_estimator.py.

Usage
-----
  python3 build_hazard_dataset.py --ckpt runs/cosine/best.pt --data_dir data_out \
      --samples_glob "runs/*/samples_cosine.txt" \
      --n_contexts 800 --k_rollouts 6 --horizon 20 --temperature 0.10 \
      --out hazard_dataset.json
"""

import os, sys, json, glob, argparse, random, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from src.utils import set_seed, load_json
from src.model import CharTransformerLM
from src.decoding import top_p_filtering

# Reuse the FROZEN, whitespace-normalized event definition -- the hazard
# labels must be computed with the exact same event the rest of the paper
# uses, or the estimator would be trained against a different target.
from sampling_eval import persistent_loop_onset


# ---- Cheap features (no model forward pass needed beyond what decoding
# already does; these mirror the signals RecurrenceRiskDecoder already
# tracks internally) ----

def rep_rate(window_text, n=5):
    if len(window_text) < n:
        return 0.0
    grams = [window_text[i:i+n] for i in range(len(window_text)-n+1)]
    from collections import Counter
    c = Counter(grams)
    return sum(v-1 for v in c.values() if v > 1) / len(grams)


def char_entropy(window_text):
    if not window_text:
        return 0.0
    from collections import Counter
    import math
    c = Counter(window_text); t = len(window_text)
    return -sum((v/t)*math.log2(v/t) for v in c.values())


def longest_suffix_match(ids, max_history=200):
    """LZ-style: length of the longest trailing run that recurs earlier
    in the recent history. O(window) via simple scan, no model call."""
    h = ids[-max_history:] if len(ids) > max_history else ids
    n = len(h)
    if n < 2:
        return 0
    best = 0
    for start in range(n - 1):
        length = 0
        while (start + length < n - 1 and
               h[start + length] == h[n - 1 - length] and
               length < n - start - 1):
            length += 1
        best = max(best, length)
    return best


def exact_risk_signal(ids, n_min=3, n_max=6):
    """Same recurrence-risk indicator used elsewhere: fraction of orders in
    [n_min,n_max] for which the CURRENT trailing context already reproduces
    an earlier n-gram (i.e. the risk of the token that was JUST placed)."""
    hits, total = 0, 0
    L = len(ids)
    for n in range(n_min, n_max + 1):
        if L < n:
            continue
        total += 1
        ctx = tuple(ids[L-n:L])
        # does this exact n-gram occur earlier in the sequence?
        for i in range(L - n):
            if tuple(ids[i:i+n]) == ctx:
                hits += 1
                break
    return hits / total if total else 0.0


def compute_features(ids, itos, window=100):
    text_window = "".join(itos.get(str(int(i)), "?") for i in ids[-window:])
    return {
        "rep_rate_5":      rep_rate(text_window, 5),
        "entropy":         char_entropy(text_window),
        "exact_risk":      exact_risk_signal(ids),
        "lookback_match":  longest_suffix_match(ids) / 60.0,  # normalised
    }


def _extract_labeled_texts(samples_globs, strategy_filter=None):
    """Parse sample files into (strategy_name, text) pairs, same header-
    matching logic as before, factored out so it can be reused."""
    header_re = re.compile(r"^\[([a-z0-9_.]+)\]\[[a-z0-9_.]+\]\s*prompt=", re.IGNORECASE)
    out = []
    for pattern in samples_globs:
        for path in glob.glob(pattern):
            txt = open(path, encoding="utf-8", errors="ignore").read()
            entries = re.split(r"\n-{10,}\n", txt)
            headers = re.findall(r"^\[([a-z0-9_.]+)\]\[[a-z0-9_.]+\]\s*prompt=.*$",
                                  txt, re.MULTILINE)
            hi = 0
            for b in entries:
                cut_at = b.find("\n\n[")
                if cut_at >= 0:
                    b = b[:cut_at]
                b = b.strip()
                if header_re.match(b):
                    continue
                strat_name = headers[hi] if hi < len(headers) else None
                hi += 1
                if strategy_filter and (strat_name is None or not any(
                        s in strat_name for s in strategy_filter)):
                    continue
                if b:
                    out.append((strat_name, b))
    return out


def sample_contexts_from_files(samples_globs, itos_inv, n_contexts, min_ctx=40,
                                max_ctx=460, seed=0, strategy_filter=None,
                                near_onset_frac=0.7, near_onset_window=100):
    """
    Mix of two kinds of contexts, NOT uniform random cut points:

    (1) NEAR-ONSET contexts (near_onset_frac of the total): for any saved
        text where the frozen persistent_loop_onset event actually fires at
        position P, cut points are drawn from [max(min_ctx, P-window), P] --
        i.e. specifically the run-up to a real loop, which is where a
        hazard estimator needs positive signal. This is necessary because
        even a "risky" decoder's samples spend MOST of their length in
        ordinary, not-about-to-loop text (e.g. temp=0.1 only loops in ~30%
        of full 500-char samples): a uniformly random cut point is very
        likely to land far from the one place risk was actually elevated.

    (2) SAFE contexts (the remainder): cut points drawn uniformly at random,
        as before, giving negative examples so the estimator isn't trained
        on positives alone.

    strategy_filter: as before, restricts which decoders' output is used
    as a source at all (defaults set by the caller to risk-prone decoders).
    """
    from sampling_eval import persistent_loop_onset
    rng = random.Random(seed)
    labeled = _extract_labeled_texts(samples_globs, strategy_filter)
    if not labeled:
        return []

    near_onset_pool = []   # (text, onset_pos)
    safe_pool = []         # text
    for strat, text in labeled:
        if len(text) <= min_ctx:
            continue
        onset = persistent_loop_onset(text)
        if onset >= min_ctx:
            near_onset_pool.append((text, onset))
        else:
            safe_pool.append(text)

    print(f"  context source stats: {len(labeled)} blocks parsed, "
          f"{len(near_onset_pool)} contain a detected onset (near-onset pool), "
          f"{len(safe_pool)} do not (safe pool)")

    n_near = int(round(n_contexts * near_onset_frac)) if near_onset_pool else 0
    n_safe = n_contexts - n_near

    contexts = []
    for _ in range(n_near):
        text, onset = rng.choice(near_onset_pool)
        lo = max(min_ctx, onset - near_onset_window)
        hi = max(lo + 1, min(onset, len(text)))
        cut = rng.randint(lo, hi)
        contexts.append(text[:cut])
    for _ in range(n_safe):
        if not safe_pool and not near_onset_pool:
            break
        pool_texts = safe_pool if safe_pool else [t for t, _ in near_onset_pool]
        t = rng.choice(pool_texts)
        if len(t) <= min_ctx:
            continue
        cut = rng.randint(min_ctx, min(max_ctx, len(t)))
        contexts.append(t[:cut])

    rng.shuffle(contexts)
    return contexts


@torch.no_grad()
@torch.no_grad()
def select_candidates(model, stoi, itos, context_ids, device, n_top=8, n_random=4, seed=0):
    """
    Given a context, return a set of candidate next-tokens to probe: the
    n_top highest-probability tokens under the model's own distribution
    (these are what actually matter for the projection, since exp(-lambda*
    hazard) reweights THEIR mass) plus n_random additional tokens sampled
    from the remaining probability mass, so the training set also covers
    lower-probability but still plausible candidates.
    """
    rng = random.Random(seed)
    idx = torch.tensor([context_ids[-model.block_size:]], dtype=torch.long, device=device)
    logits, _ = model(idx)
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    order = torch.argsort(probs, descending=True).tolist()
    top = order[:n_top]
    rest = order[n_top:]
    rand_extra = rng.sample(rest, min(n_random, len(rest))) if rest else []
    return top + rand_extra, probs.tolist()


@torch.no_grad()
def batched_rollout_labels_per_candidate(model, stoi, itos, candidates_per_context,
                                         k_rollouts, horizon, temperature, device,
                                         batch_size=256):
    """
    For each (context, candidate_token) pair, force the candidate as the
    FIRST generated token, then continue k_rollouts independent stochastic
    continuations for the remaining (horizon-1) steps under the reference
    temperature. This is the per-candidate hazard the projection actually
    needs -- "if I choose v now, what is P(loop within the horizon)" --
    matching the structure validated on the synthetic FSM, not an average
    over whatever the model would have chosen on its own.

    candidates_per_context: list of (context_ids, candidate_token_list,
    full_prob_vector) triples, one per context, as built by select_candidates.

    Returns a flat list of examples: one per (context, candidate) pair, with
    the empirical hazard label and the candidate identity/probability, so
    compute_features() can be called per-example afterward.
    """
    unk = stoi.get(" ", 0)
    block_size = model.block_size

    flat_ids = []
    owner = []
    examples_meta = []

    for ci, (ctx_ids, cand_list, probs) in enumerate(candidates_per_context):
        for cand in cand_list:
            ex_idx = len(examples_meta)
            examples_meta.append((ci, cand, probs[cand]))
            for _ in range(k_rollouts):
                flat_ids.append(list(ctx_ids) + [cand])
                owner.append(ex_idx)

    hits = [0] * len(examples_meta)
    total = [0] * len(examples_meta)

    for start in range(0, len(flat_ids), batch_size):
        chunk = flat_ids[start:start+batch_size]
        chunk_owner = owner[start:start+batch_size]
        max_len = max(len(c) for c in chunk)
        padded = [[unk]*(max_len-len(c)) + c for c in chunk]
        idx = torch.tensor(padded, dtype=torch.long, device=device)
        ctx_len = max_len  # includes the forced candidate token already

        for _ in range(horizon - 1):   # candidate already consumed 1 step
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs_step = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs_step, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        for row, ex_idx in zip(idx.tolist(), chunk_owner):
            gen_part = row[ctx_len - 1:]  # candidate token + everything after
            text = "".join(itos.get(str(int(t)), "?") for t in gen_part)
            onset = persistent_loop_onset(text)
            total[ex_idx] += 1
            if onset >= 0:
                hits[ex_idx] += 1

        print(f"  rollout batch {start}-{start+len(chunk)} / {len(flat_ids)} done")

    out = []
    for (ci, cand, p), h, t in zip(examples_meta, hits, total):
        out.append({"context_idx": ci, "candidate_token": cand,
                    "candidate_model_prob": p,
                    "hazard_label": (h/t if t else float("nan"))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/cosine/best.pt")
    ap.add_argument("--data_dir", default="data_out")
    ap.add_argument("--samples_glob", nargs="*",
                     default=["runs/*/samples_cosine.txt"])
    ap.add_argument("--n_contexts", type=int, default=150,
                     help="Reduced from the context-only design's 800, since "
                          "each context now expands into n_top+n_random "
                          "candidate examples, each with its own rollouts -- "
                          "the real unit of data is (context, candidate).")
    ap.add_argument("--n_top_candidates", type=int, default=8,
                     help="Highest-probability next-tokens to probe per "
                          "context -- these are what actually matter for the "
                          "projection, since it reweights the model's own "
                          "probability mass.")
    ap.add_argument("--n_random_candidates", type=int, default=4,
                     help="Additional lower-probability tokens to probe per "
                          "context, for feature-space coverage beyond the "
                          "top-probability set.")
    ap.add_argument("--strategy_filter", nargs="*",
                     default=["greedy", "temp_0.05", "temp_0.1", "rep_penalty"],
                     help="Only draw contexts from samples whose strategy name "
                          "contains one of these substrings (matched against "
                          "the [name] in each sample's header). Defaults to "
                          "the risk-prone configurations (greedy, low "
                          "temperature, rep_penalty) so the context pool isn't "
                          "diluted by decoders (dual/adaptive/lookback/"
                          "no_repeat) that were specifically built to prevent "
                          "the event we're trying to learn to predict. Pass "
                          "--strategy_filter (with no values) to disable "
                          "filtering and use every available sample.")
    ap.add_argument("--k_rollouts", type=int, default=4,
                     help="Independent rollouts per (context, candidate) pair. "
                          "Reduced from 6 since the candidate dimension already "
                          "multiplies total rollout volume substantially.")
    ap.add_argument("--horizon", type=int, default=150,
                     help="Rollout length in characters. MUST be well above "
                          "the event's minimum detectable length: R=5 exact "
                          "repeats of a ~15-20 char phrase period need "
                          "75-100+ characters at minimum, and observed onsets "
                          "for even the most degenerate (greedy) generations "
                          "were ~90-140 chars in prior runs. A horizon of 20 "
                          "structurally CANNOT ever observe this event -- "
                          "every rollout would report no-loop regardless of "
                          "how degenerate the continuation is, producing a "
                          "trivially all-zero label set that looks like a "
                          "sampling problem but is actually a horizon bug.")
    ap.add_argument("--temperature", type=float, default=0.10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out", default="hazard_dataset.json")
    args = ap.parse_args()

    device = (torch.device("mps") if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = CharTransformerLM(
        vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
        n_layer=cfg["n_layer"], n_embd=cfg["n_embd"],
        n_head=cfg["n_head"], dropout=0.0).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    vocab = load_json(os.path.join(args.data_dir, "vocab.json"))
    stoi, itos = vocab["stoi"], vocab["itos"]

    print(f"Collecting contexts from: {args.samples_glob}")
    contexts = sample_contexts_from_files(args.samples_glob, itos, args.n_contexts,
                                           strategy_filter=args.strategy_filter)
    print(f"  (strategy_filter={args.strategy_filter or 'disabled -- using all samples'})")
    print(f"Got {len(contexts)} contexts.")
    if not contexts:
        print("No contexts found -- check --samples_glob paths. Aborting.")
        return

    n_examples_total = len(contexts) * (args.n_top_candidates + args.n_random_candidates)
    n_rollouts_total = n_examples_total * args.k_rollouts
    print(f"Selecting ~{args.n_top_candidates + args.n_random_candidates} candidates per "
          f"context -> ~{n_examples_total} (context,candidate) examples "
          f"x {args.k_rollouts} rollouts = ~{n_rollouts_total} rollouts x "
          f"{args.horizon} steps (batched at {args.batch_size} per step)")

    # Pre-flight sanity check: the persistent-loop event needs at least
    # min_p*R characters to EVER fire (R=5, min_p=2 by default -> 10 chars
    # minimum), and in practice needs far more (~90-140 chars observed for
    # even the most degenerate generations in prior runs). A horizon below
    # that floor guarantees a degenerate all-zero label set no matter how
    # risky the contexts/candidates are -- catch this BEFORE spending budget.
    MIN_SANE_HORIZON = 80
    if args.horizon < MIN_SANE_HORIZON:
        print(f"\nWARNING: --horizon={args.horizon} is below {MIN_SANE_HORIZON}, "
              f"the approximate minimum length the persistent-loop event has "
              f"actually needed to fire in this project's prior runs. This will "
              f"very likely produce an all-zero label set regardless of "
              f"context/candidate choice. Proceeding anyway since you set it "
              f"explicitly.\n")

    print("\nSelecting candidate next-tokens per context (one forward pass each)...")
    context_id_lists = []
    candidates_per_context = []
    for ctx in contexts:
        ids = [stoi.get(c, stoi.get(" ", 0)) for c in ctx]
        cands, probs = select_candidates(model, stoi, itos, ids, device,
                                          n_top=args.n_top_candidates,
                                          n_random=args.n_random_candidates,
                                          seed=hash(ctx) % 10000)
        context_id_lists.append(ids)
        candidates_per_context.append((ids, cands, probs))

    print("\nRunning per-candidate batched rollouts...")
    examples = batched_rollout_labels_per_candidate(
        model, stoi, itos, candidates_per_context,
        args.k_rollouts, args.horizon, args.temperature, device, args.batch_size)

    print("\nComputing features per (context, candidate) example...")
    dataset = []
    for ex in examples:
        ci = ex["context_idx"]
        ctx_ids = context_id_lists[ci]
        # Context-level features (as before) plus this candidate's own
        # immediate exact-risk contribution and the model's own probability
        # for it -- the per-candidate signals a real decoder has available.
        ctx_feats = compute_features(ctx_ids, itos)
        cand = ex["candidate_token"]
        extended_ids = ctx_ids + [cand]
        cand_immediate_risk = exact_risk_signal(extended_ids)
        dataset.append({
            **ctx_feats,
            "candidate_model_prob": ex["candidate_model_prob"],
            "candidate_immediate_risk": cand_immediate_risk,
            "hazard_label": ex["hazard_label"],
            "context_len": len(ctx_ids),
        })

    with open(args.out, "w") as f:
        json.dump(dataset, f, indent=2)

    labs = [d["hazard_label"] for d in dataset if d["hazard_label"] == d["hazard_label"]]
    print(f"\nSaved {len(dataset)} (context, candidate) examples -> {args.out}")
    print(f"Label distribution: mean={np.mean(labs):.3f}  "
          f"frac==0: {sum(1 for l in labs if l==0)/len(labs):.2f}  "
          f"frac==1: {sum(1 for l in labs if l==1)/len(labs):.2f}")
    # Also report the split between "immediately risky" candidates (already
    # completing a repeat) and "novel" candidates, since these SHOULD show
    # very different hazard rates if the per-candidate design is working --
    # this is the direct sanity check that per-candidate signal exists at all.
    risky = [d["hazard_label"] for d in dataset if d["candidate_immediate_risk"] > 0
             and d["hazard_label"] == d["hazard_label"]]
    novel = [d["hazard_label"] for d in dataset if d["candidate_immediate_risk"] == 0
             and d["hazard_label"] == d["hazard_label"]]
    if risky and novel:
        print(f"\nSanity check -- mean hazard by candidate type:")
        print(f"  candidates with immediate exact-risk > 0: mean hazard = {np.mean(risky):.3f}  (n={len(risky)})")
        print(f"  candidates with immediate exact-risk = 0: mean hazard = {np.mean(novel):.3f}  (n={len(novel)})")
        print(f"  -> if the risky group's mean is meaningfully higher, per-candidate signal exists.")
    if np.mean(labs) < 0.02 or np.mean(labs) > 0.98:
        print("\nWARNING: labels are still nearly degenerate even per-candidate.")
        print("Check the sanity-check split above before concluding this is a dead end --")
        print("a low OVERALL mean with a clear risky-vs-novel gap is still a usable signal.")


if __name__ == "__main__":
    main()