# Recurrence-Risk Decoding for Repetition-Loop Control

A decoding-time method that suppresses repetition loops in autoregressive
language models by penalising candidate tokens in proportion to the risk that
they would extend an already-repeated n-gram. Derived as the exact
minimum-distortion (KL-projection) solution to a risk-bounded decoding
problem:

    minimize  KL(q || p)   subject to   E_q[risk] <= eps,   q in simplex

whose Lagrangian dual has the exponential form
`q(v) ∝ p(v) * exp(-lambda * risk(v))` — exact **only** when `lambda` is
solved from `eps` at every step (`mode="dual"`), not when it is a hand-set
constant (`mode="fixed"`) or an online heuristic (`mode="adaptive"`). See
[last_paper.tex](last_paper.tex) for the full derivation, all experiments,
and — just as importantly — the explicit list of what is *not* claimed.

This README describes how to reproduce every experiment. It intentionally
does **not** embed result tables/numbers inline (an earlier version of this
file did, and they went stale every time a run was redone) — current numbers
live in `runs/<out_dir>/metrics_*.csv` and are quoted, with confidence
intervals, in the paper.

## Project structure

```
src/
  model.py                    CharTransformerLM (GPT-style decoder-only Transformer)
  dataset.py                  CharBinDataset (binary file reader)
  decoding.py                 All decoders: baselines + RecurrenceRiskDecoder
  utils.py                    set_seed, bpc_from_loss, save/load JSON

prepare.py                    Download and preprocess Gutenberg books
train.py / train_cosine.py    Training (constant LR / cosine LR + warmup)
sample.py                     Single-prompt text generation
eval_bpc.py                   Full-book BPC with paired block bootstrap

sampling_eval.py              Main decoding-strategy comparison (character model):
                               equal-budget sweeps, multiple loop definitions,
                               survival analysis, prompt-clustered paired
                               bootstrap, per-step diagnostics (opt-in).
validate_loop_event_v2.py     Calibrate/freeze/validate the character-level
                               persistent-loop event on corpora disjoint from
                               every headline-experiment corpus.
validate_loop_event_subword.py
                               Same calibrate/freeze/validate protocol, token-level,
                               for a pretrained subword model's own tokenizer.
pretrained_subword_pilot.py   Cheap pilot: confirms a measurable low-temperature
                               repetition regime exists on a pretrained subword
                               model BEFORE committing to a full comparison there.
gpt2_full_comparison.py       Full decoder comparison on a pretrained subword
                               model (GPT-2 by default), reusing the SAME
                               decoder classes from src/decoding.py.
generalize_gpt2.py            Earlier, narrower subword generalization check
                               (superseded by gpt2_full_comparison.py for the
                               headline comparison; kept for its own record).

synthetic_fsm_hazard.py       Synthetic finite-state hazard experiment with an
                               exactly DP-computed hazard (ground truth known).
build_hazard_dataset.py       Builds the real-model hazard-transfer dataset.
test_multistep_hazard.py      Multi-step hazard probe on the real model.
compute_significance.py       Exact prompt-level significance calculations
                               (replaces ad-hoc inline recomputation).
plot_results.py / plot_training.py
                               Figures (Pareto, KM survival, loop-robustness,
                               ablations / training curves).

human_eval_tool.html          Self-contained, offline blinded human-evaluation
                               tool (loop-detection agreement + pairwise
                               preference). Open directly in a browser, no
                               server or internet needed.
aggregate_human_eval.py       Aggregates one or more completed human_eval_tool.html
                               result JSONs into agreement/preference statistics.
```

## 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`transformers` (needed only for the pretrained-subword-model scripts) is
included in `requirements.txt`; the character-model pipeline never imports it.

## 2. Data preparation

```bash
python3 prepare.py \
  --out_dir data_out \
  --book_ids 1342 84 1661 98 \
  --val_books 1 --test_books 1 \
  --max_chars_per_book 2000000
```

Split is at **book level** (train/val/test from different authors), so
evaluation tests cross-author generalisation, not within-book memorisation.
The non-fiction generalization corpus (`data_out_nonfiction/`) is built the
same way from a disjoint set of books; see `runs/nonfiction_baseline/config.json`
for the exact book IDs used.

## 3. Training

```bash
# Sanity check — expect BPC below 1.0 within 3,000 steps
python3 train.py --data_dir data_out --out_dir runs/overfit \
  --overfit_chars 50000 --max_steps 3000

# Two independently-trained checkpoints of the primary model (used later as
# the "second independent checkpoint" generalization check):
python3 train.py --data_dir data_out --out_dir runs/baseline --max_steps 20000
python3 train_cosine.py --data_dir data_out --out_dir runs/cosine \
  --max_steps 20000 --warmup_steps 500
```

## 4. BPC evaluation

```bash
python3 eval_bpc.py --ckpt runs/baseline/best.pt --split test --mode bootstrap
python3 eval_bpc.py --ckpt runs/baseline/best.pt runs/cosine/best.pt \
  --split test --mode paired
```

## 5. Loop-event validation (must run before trusting any loop-rate number)

The persistent-cycle event's (R, period_max) thresholds are calibrated and
frozen on corpora disjoint from every corpus used in the headline
experiments, then validated for false-positive rate on a second, still-disjoint
battery. Re-run this whenever the event definition, corpus, or tokenizer
changes — do not reuse a frozen threshold across tokenizers without
re-validating (this project was burned by that exact assumption once; see
`validate_loop_event_subword.py`'s docstring).

```bash
python3 validate_loop_event_v2.py --out_dir loop_event_v2_report
```

## 6. Main decoding-strategy comparison (character model)

```bash
python3 sampling_eval.py \
  --ckpt runs/baseline/best.pt runs/cosine/best.pt \
  --out_dir runs/main_comparison \
  --n_seeds 10 --n_chars 500 --prompt_set test
```

`--prompt_set test` uses the 15 prompts never inspected while tuning
hyperparameters; headline numbers must use `test`, not `dev`.

Decoders compared (see `src/decoding.py` docstrings for full detail on each):

| Name in configs | Class | Note |
|---|---|---|
| `lt_temp_*`, `lt_rep_penalty_*` | — | Standard baselines |
| `lt_no_repeat_Ngram` | — | Hard constraint; kept separate, directly forbids the measured event |
| `lt_suffixmatch_*` | `SuffixMatchDecoder` | Homemade LZ77-style baseline. **Not** the published Look-back algorithm (renamed from the earlier, misleading `LookBackDecoder`) |
| `lt_fsd` | `FSDDecoder` | Good-faith FSD-*style* reconstruction, not a certified reproduction |
| `lt_lzpenalty` | `LZPenaltyDecoder` | Authentic reimplementation of Ginart et al., *LZ Penalty* (arXiv:2504.20131, TMLR 2026) |
| `lt_risk_only`, `lt_adaptive` | `RecurrenceRiskDecoder` (`mode="fixed"` / `"adaptive"`) | No exact projection guarantee |
| `lt_dual_eps*` | `RecurrenceRiskDecoder` (`mode="dual"`) | The **only** mode entitled to the exact minimum-distortion claim; lambda is solved from eps every step in log-space (see `_solve_dual_lambda`) |

To also save full per-step dual-solver traces (lambda, achieved risk,
feasible/structurally-infeasible, KL, entropy at every decoding step, not
just the per-sample mean — needed to audit the solver, not just its summary)
for a handful of samples per dual-mode config:

```bash
python3 sampling_eval.py \
  --ckpt runs/cosine/best.pt --out_dir runs/main_comparison \
  --n_seeds 10 --n_chars 500 --prompt_set test \
  --only lt_risk_only lt_adaptive lt_dual_eps0.01 lt_dual_eps0.05 \
  --save_per_step_diagnostics 5
```
Writes `runs/main_comparison/per_step_diagnostics/*.jsonl`. This is a
short, four-config rerun, not a full 13-config one — the other configs
never populate `per_step_log` and are unaffected by the dual-mode fix.

## 7. Generalization checks

**Non-fiction corpus, same architecture** (same protocol, disjoint corpus):
```bash
python3 sampling_eval.py \
  --ckpt runs/nonfiction_baseline/best.pt --data_dir data_out_nonfiction \
  --out_dir runs/nonfiction_comparison \
  --n_seeds 10 --n_chars 500 --prompt_set test
```

**Pretrained subword model (GPT-2)** — pilot, then calibration, then full
comparison, in that order; do not skip the pilot or reuse the character-level
event thresholds (see script docstrings for why):
```bash
python3 pretrained_subword_pilot.py --model gpt2 --n_prompts 5 --n_seeds 5
# -> confirms a measurable low-temperature repetition regime exists at all

python3 validate_loop_event_subword.py --model gpt2 --out_dir loop_event_subword_report
# -> freezes GPT-2's own (R, period_max), does not reuse the char-level one

python3 gpt2_full_comparison.py --model gpt2 --n_seeds 10 --n_tokens 200 \
  --out_dir runs/gpt2_full_comparison
```

## 8. Hazard-aware extension (synthetic-validated, real-model transfer tested honestly)

```bash
python3 synthetic_fsm_hazard.py          # exact DP ground truth
python3 build_hazard_dataset.py          # real-model hazard-transfer dataset
python3 test_multistep_hazard.py         # multi-step hazard probe on the real model
```
Validated where ground truth is exactly known; found not to transfer to the
real model under the specific rollout estimator tested. Reported in the
paper as a scope-limited null result of that estimator, not a general claim.

## 9. Significance testing and figures

```bash
python3 compute_significance.py
python3 plot_results.py --dir runs/main_comparison --out runs/main_comparison/plots
```

## 10. Human evaluation

`human_eval_tool.html` is a single self-contained file — open it directly in
any browser (`file://...`, no server, no internet, works fully offline).
It runs two blinded tasks (loop-detection agreement on real generated
passages; pairwise preference on `dual` vs `suffixmatch` outputs the
automatic metrics alone cannot resolve), randomizes order and left/right
per participant, and ends with a "copy results" step that produces one JSON
blob per participant. Send each participant's JSON blob back and aggregate:

```bash
python3 aggregate_human_eval.py results_participant1.json results_participant2.json ...
```

See `last_paper.tex`, Section "Human evaluation", for the full protocol.

## 11. Full reproduction, start to finish

```bash
pip install -r requirements.txt
python3 prepare.py --out_dir data_out --book_ids 1342 84 1661 98 --val_books 1 --test_books 1
python3 train.py --data_dir data_out --out_dir runs/baseline --max_steps 20000
python3 train_cosine.py --data_dir data_out --out_dir runs/cosine --max_steps 20000 --warmup_steps 500
python3 validate_loop_event_v2.py --out_dir loop_event_v2_report
python3 sampling_eval.py --ckpt runs/baseline/best.pt runs/cosine/best.pt \
  --out_dir runs/main_comparison --n_seeds 10 --n_chars 500 --prompt_set test
python3 compute_significance.py
python3 plot_results.py --dir runs/main_comparison --out runs/main_comparison/plots
```

## Hardware & runtime (approximate — varies by machine)

| Hardware | Training speed | Full training (20k steps) |
|----------|-----------------|----------------------------|
| Apple M-series (MPS) | ~26 it/s | ~13 min |
| CPU only | ~8 it/s | ~40 min |
| CUDA GPU | ~80 it/s | ~4 min |

The full `sampling_eval.py` comparison (13 configs × up to 15 prompts × 10
seeds × 500 chars, plus per-sample mcNLL) is the long-running step —
budget on the order of a few hours on CPU/MPS, dominated by the
model-consistency-NLL pass, not by decoding itself.

## What this project does and does not claim

Kept intentionally short here to avoid drifting out of sync with the paper;
the authoritative, current version is `last_paper.tex`'s Discussion and
"What is not claimed" list. In brief: the theorem guarantees a **per-step**
bound on expected recurrence risk when the dual projection numerically
succeeds; it says nothing about the sequence-level count of persistent
loops. Zero observed loops across evaluated samples is reported throughout
as a replicated *empirical* finding, never as something the theorem
predicts or as a guarantee that "transfers."
