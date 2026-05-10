# Sampling Effects in Character-Level Text Generation

A GPT-style character-level Transformer trained on Project Gutenberg novels.
Studies decoding strategies and introduces a **recurrence-aware adaptive decoder**
that softly penalises candidates which would extend repeated n-grams, adapting
penalty strength online based on recent repetition rate and entropy.

---

## Project Structure

```
.
├── src/
│   ├── model.py          # CharTransformerLM (GPT-style decoder-only Transformer)
│   ├── dataset.py        # CharBinDataset (binary file reader)
│   ├── decoding.py       # generate() with all strategies + RecurrenceAwareDecoder
│   └── utils.py          # set_seed, bpc_from_loss, save/load JSON
│
├── prepare.py            # Download and preprocess Gutenberg books
├── train.py              # Baseline training (constant LR)
├── train_cosine.py       # Cosine LR training (warmup + cosine decay)
├── sample.py             # Single-prompt text generation
├── eval_bpc.py           # Full-book BPC evaluation with paired bootstrap test
├── sampling_eval.py      # Decoding strategy evaluation with quality metrics
├── plot_training.py      # Training curve plots from log.csv
├── demo.py               # Quick demo — no training required
│
├── requirements.txt
└── README.md
```

---

## 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Data Preparation

```bash
python3 prepare.py \
  --out_dir data_out \
  --book_ids 1342 84 1661 98 \
  --val_books 1 \
  --test_books 1 \
  --max_chars_per_book 2000000
```

| Book ID | Title | Split |
|---------|-------|-------|
| 1342 | Pride and Prejudice (Austen) | Train |
| 84 | Frankenstein (Shelley) | Train |
| 1661 | Adventures of Sherlock Holmes (Doyle) | Validation |
| 98 | A Tale of Two Cities (Dickens) | Test |

Split is at **book level** — train, val, and test come from different authors,
testing genuine cross-author generalisation rather than within-book memorisation.

---

## 3. Sanity Check

```bash
python3 train.py \
  --data_dir data_out \
  --out_dir runs/overfit \
  --overfit_chars 50000 \
  --max_steps 3000
```

Expected: BPC below 1.0 within 3,000 steps.

---

## 4. Full Training

### Experiment 1 — Baseline (constant LR)

```bash
python3 train.py \
  --data_dir data_out \
  --out_dir runs/baseline \
  --max_steps 20000
```

### Experiment 2 — Cosine LR Schedule

```bash
python3 train_cosine.py \
  --data_dir data_out \
  --out_dir runs/cosine \
  --max_steps 20000 \
  --warmup_steps 500
```

---

## 5. BPC Evaluation (Corrected Protocol)

> **Important:** The original `estimate_loss()` in `train.py` sampled only
> 50 random overlapping batches, covering ~27% of the test book.
> This **inflated BPC by ~1.46 bits**. The numbers below are from the
> corrected protocol only.
>
> ~~Baseline test BPC: 3.702~~ — **OUTDATED, do not cite**
> ~~Cosine test BPC: 3.530~~ — **OUTDATED, do not cite**

`eval_bpc.py` tiles the entire book with non-overlapping windows and uses
block bootstrap (n=2000, block_len=512) for confidence intervals.

### Full-book eval with 95% CI

```bash
python3 eval_bpc.py \
  --ckpt runs/baseline/best.pt \
  --split test --mode bootstrap
```

### Paired bootstrap significance test

```bash
python3 eval_bpc.py \
  --ckpt runs/baseline/best.pt runs/cosine/best.pt \
  --split test --mode paired
```

The paired test operates on per-token NLL differences, cancelling
position-level variance common to both models and giving higher statistical
power than comparing two independent CIs.

### Corrected BPC results

| Checkpoint | Val BPC | Test BPC | Test 95% CI |
|------------|---------|----------|-------------|
| Baseline (const LR) | 2.179 | 2.238 | [2.225, 2.250] |
| Cosine LR + warmup  | —     | 2.234 | [2.220, 2.246] |

**Paired bootstrap:** delta BPC = +0.0046 (A−B), 95% CI [+0.002, +0.007],
p = 0.0005. The cosine schedule is statistically significantly better,
though the effect size is small (0.005 BPC). Independent CIs overlap
because they do not account for the strong per-token correlation between
models — the paired test has higher power and is the correct comparison.

---

## 6. Generate Text

```bash
python3 sample.py \
  --ckpt runs/cosine/best.pt \
  --prompt "CHAPTER 1\n" \
  --temperature 0.8 \
  --max_new_chars 800
```

---

## 7. Decoding Strategy Evaluation

Evaluates all strategies on **both** checkpoints.
10 seeds × 5 prompts = 50 samples per strategy per checkpoint.

```bash
python3 sampling_eval.py \
  --ckpt runs/baseline/best.pt runs/cosine/best.pt \
  --out_dir runs/sampling_eval_v3 \
  --n_seeds 10 --n_chars 500
```

### Strategies

| Strategy | Category | Description |
|----------|----------|-------------|
| Greedy | baseline | Deterministic; always degenerates |
| Temperature 0.8 | baseline | Standard stochastic sampling |
| Nucleus p=0.95 | baseline | Holtzman et al., 2020 |
| Typical p=0.9 | probabilistic | Meister et al., 2023 |
| Rep. penalty 1.3 | probabilistic | Keskar et al., 2019 |
| Mirostat τ=5 | probabilistic | Basu et al., 2021 |
| No-repeat 4-gram | hard constraint | See note below |
| **Adaptive** | **novel** | **RecurrenceAwareDecoder — see below** |

> **Note on no-repeat 4-gram:** This strategy directly forbids any token
> that would create a repeated 4-gram. It trivially achieves rep_rate=0
> because it mechanically prevents the metric from occurring — not because
> of any improvement in generation quality. It is reported as a hard
> constraint baseline and must not be ranked against probabilistic methods.

### Recurrence-Aware Adaptive Decoder

The novel contribution of this project. At each generation step:

1. **Per-candidate risk score** `r(v)`: fraction of n-gram sizes in {3..6}
   for which appending token `v` to the recent context would create a
   repeated suffix. This is computed before sampling.

2. **Soft penalty**: subtract `alpha × r(v)` from the logit of each
   candidate. Unlike no-repeat-ngram, this is a soft discouragement,
   not a hard ban — low-risk tokens are unaffected.

3. **Online adaptation**: `alpha` is updated at each step based on
   recent repetition rate (increases penalty when rep is high) and
   recent entropy (decreases penalty when generation is healthy).
   This allows the decoder to react to degeneration before it fully sets in.

### Degeneration metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| TTR | ↑ | Type-token ratio |
| 4-gram entropy | ↑ | Shannon entropy of character 4-grams |
| Rep. rate (n=5) | ↓ | Fraction of repeated 5-grams |
| Rep. n-gram mass (n=2,4,6) | ↓ | Fraction of n-gram occurrences that repeat |
| Loop onset | ↑ | First position where a length-10 n-gram repeats |
| Longest rep. substring | ↓ | Length of longest repeated substring |
| Entropy trajectory | — | 4-gram entropy in 4 equal windows |

### Quality metrics (new)

| Metric | Direction | Description |
|--------|-----------|-------------|
| Gen. NLL (BPC) | ↓ | NLL of generated text under the model — proxy for staying in-distribution |
| N-gram sim. (4) | ↑ | 1 − JSD between generated and val.bin 4-gram distributions |
| Survival AUC | ↑ | Area under Kaplan-Meier loop-onset survival curve (censored correctly) |

### Results (cosine checkpoint, 50 samples per strategy)

| Strategy | TTR ↑ | RepRate ↓ | NLL ↓ | Sim ↑ | SAUC ↑ | Censored |
|----------|--------|-----------|-------|-------|--------|----------|
| Greedy | 0.117 | 0.877 | **0.740** | 0.075 | 0.055 | 0/50 |
| Temp. 0.8 | 0.730 | 0.074 | 1.411 | 0.247 | 0.807 | 32/50 |
| Nucleus p=0.95 | 0.753 | 0.065 | 1.381 | 0.248 | 0.786 | 28/50 |
| Typical p=0.9 | 0.730 | 0.068 | 1.392 | 0.251 | 0.803 | 28/50 |
| Rep. penalty 1.3 | **0.838** | **0.033** | 1.693 | 0.218 | 0.931 | 39/50 |
| Mirostat τ=5 | 0.804 | 0.046 | 1.634 | 0.230 | 0.839 | 33/50 |
| No-repeat 4-gram* | 0.946 | 0.000 | 1.648 | 0.233 | 1.000 | 50/50 |
| **Adaptive** | 0.746 | 0.041 | **1.273** | **0.268** | **0.953** | **45/50** |

*Hard constraint — not comparable to probabilistic methods.

**Greedy NLL is lowest (0.740) because the model is highly confident in its
repetition loops — low NLL does not imply quality when the model has degenerated.**

### Key finding

The recurrence-aware adaptive decoder achieves the best generated-text NLL
(1.273 BPC) and n-gram distributional similarity to held-out text (0.268)
among all probabilistic strategies, and the highest survival AUC (0.953)
outside the hard no-repeat constraint. This comes at a modest cost in TTR
and repetition rate compared to fixed repetition penalty, suggesting that
soft online penalisation of risky candidates preserves generation quality
better than global token-level penalties.

The trade-off is honest: adaptive is not better on every metric. Rep. penalty
achieves lower repetition rate (0.033 vs 0.041) and higher TTR (0.838 vs 0.746).
The adaptive decoder's advantage is in quality metrics — NLL and distributional
similarity — which measure whether the generated text stays close to the
model's learned distribution and to real text.

---

## 8. Hardware & Runtime

| Hardware | Speed | Full training (20k steps) |
|----------|-------|--------------------------|
| Apple M-series (MPS) | ~26 it/s | ~13 minutes |
| CPU only | ~8 it/s | ~40 minutes |
| CUDA GPU | ~80 it/s | ~4 minutes |