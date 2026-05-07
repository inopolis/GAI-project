# Sampling Effects in Character-Level Text Generation

A GPT-style character-level Transformer trained on Project Gutenberg novels.
Explores decoding strategies (greedy, temperature, top-k, nucleus, typical
sampling, repetition penalty, Mirostat) and learning rate schedules.

---

## Project Structure

```
.
├── src/
│   ├── model.py          # CharTransformerLM (GPT-style decoder-only Transformer)
│   ├── dataset.py        # CharBinDataset (binary file reader)
│   ├── decoding.py       # generate() — temperature / top-k / top-p /
│   │                     #   typical_p / rep_penalty / no_repeat_ngram / Mirostat
│   └── utils.py          # set_seed, bpc_from_loss, save/load JSON
│
├── prepare.py            # Download & preprocess Gutenberg books
├── train.py              # Baseline training (constant LR)
├── train_cosine.py       # Cosine LR training (warmup + cosine decay)
├── sample.py             # Single-prompt text generation
├── eval_bpc.py           # Full-book BPC evaluation with paired bootstrap test
├── sampling_eval.py      # Systematic evaluation of decoding strategies
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

Split is at **book level** — train, val, and test come from different authors.
This tests cross-author generalisation, not just within-book memorisation.

---

## 3. Sanity Check

```bash
python3 train.py \
  --data_dir data_out \
  --out_dir runs/overfit \
  --overfit_chars 50000 \
  --max_steps 3000
```

Expected: training loss drops below 0.5 (BPC < 1.0) within 3,000 steps.

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

The original `estimate_loss()` in `train.py` sampled only 50 random overlapping
batches, covering approximately 27% of the test book. This **inflated BPC by
~1.46 bits**. The old numbers should not be cited:

> ~~Baseline test BPC: 3.702~~ — **OUTDATED, do not use**
> ~~Cosine test BPC: 3.530~~ — **OUTDATED, do not use**

Use `eval_bpc.py` for all reported results. It tiles the entire book with
non-overlapping windows and uses block bootstrap (n=2000) for confidence
intervals.

### Full-book eval with 95% CI

```bash
python3 eval_bpc.py \
  --ckpt runs/baseline/best.pt \
  --split test \
  --mode bootstrap
```

### Paired bootstrap significance test (baseline vs cosine)

```bash
python3 eval_bpc.py \
  --ckpt runs/baseline/best.pt runs/cosine/best.pt \
  --split test \
  --mode paired
```

The paired test is stronger than comparing two independent CIs: it operates
on per-token NLL differences, cancelling position-level variance common to
both models and giving higher statistical power.

### Corrected results (full-book, non-overlapping windows)

| Checkpoint | Val BPC | Test BPC | Test 95% CI |
|------------|---------|----------|-------------|
| Baseline (const LR) | 2.179 | 2.238 | [2.225, 2.250] |
| Cosine LR + warmup  | —     | 2.234 | [2.220, 2.246] |

Paired bootstrap test: delta BPC = 0.004, p-value not significant at 95%.
**There is no statistically significant difference between the two schedules**
at this model size and dataset scale. The apparent 0.172 BPC improvement in
the original report was entirely a measurement artefact.

---

## 6. Generate Text

```bash
# Temperature sampling (recommended)
python3 sample.py \
  --ckpt runs/cosine/best.pt \
  --prompt "CHAPTER 1\n" \
  --temperature 0.8 \
  --max_new_chars 800

# Nucleus sampling
python3 sample.py \
  --ckpt runs/cosine/best.pt \
  --prompt "She had never " \
  --temperature 1.0 \
  --top_p 0.95

# Greedy (will degenerate — shown for comparison only)
python3 sample.py \
  --ckpt runs/baseline/best.pt \
  --prompt "CHAPTER 1\n" \
  --temperature 0.0
```

---

## 7. Decoding Strategy Evaluation

Evaluates all strategies on **both** checkpoints. Reports mean ± std across
10 seeds × 5 prompts = 50 samples per strategy.

```bash
python3 sampling_eval.py \
  --ckpt runs/baseline/best.pt runs/cosine/best.pt \
  --out_dir runs/sampling_eval_v2 \
  --n_seeds 10 \
  --n_chars 500
```

Outputs:
- `metrics_baseline.csv` / `metrics_cosine.csv` — per-checkpoint results
- `metrics_comparison.csv` — side-by-side comparison
- `all_metrics.json` — full results with all metric columns
- `samples_baseline.txt` / `samples_cosine.txt` — one sample per strategy

### Strategies evaluated

| Strategy | Category | Notes |
|----------|----------|-------|
| Greedy | baseline | Always fails — loop onset ~27 chars |
| Temperature 0.6 / 0.8 / 1.0 | baseline | |
| Top-k (k=10) | baseline | |
| Nucleus p=0.95 | baseline | |
| Typical sampling p=0.9 | probabilistic | Meister et al. 2023 |
| Repetition penalty 1.3 | probabilistic | Keskar et al. 2019 |
| Mirostat τ=3 / τ=5 | probabilistic | Basu et al. 2021 |
| No-repeat 4-gram | hard constraint | **See note below** |

> **Note on no-repeat 4-gram:** This strategy directly forbids generating any
> token that would create a repeated 4-gram. It trivially achieves rep_rate=0
> because it mechanically prevents the metric from occurring — not because of
> any improvement in the model's generation quality. It is reported separately
> and should not be ranked against probabilistic methods.

### Degeneration metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| TTR | ↑ better | Type-token ratio (unique words / total words) |
| 4-gram entropy | ↑ better | Shannon entropy over character 4-grams |
| Rep. rate (n=5) | ↓ better | Fraction of 5-grams that are repeated |
| Rep. n-gram mass (n=2,4,6) | ↓ better | Fraction of n-gram occurrences that repeat |
| Loop onset | ↑ better | First position where a length-10 n-gram repeats (-1 = none) |
| Longest rep. substring | ↓ better | Length of longest substring appearing ≥2 times |
| Entropy trajectory | — | 4-gram entropy in 4 equal windows (detects progressive degeneration) |

---

## 8. Training Plots

```bash
python3 plot_training.py \
  --log runs/baseline/log.csv \
  --out runs/baseline/plots
```

---

## 9. Hardware & Runtime

| Hardware | Speed | Full training (20k steps) |
|----------|-------|--------------------------|
| Apple M-series (MPS) | ~26 it/s | ~13 minutes |
| CPU only | ~8 it/s | ~40 minutes |
| CUDA GPU | ~80 it/s | ~4 minutes |

---

## 10. Key Results Summary

> **Important:** The BPC numbers in the table below are from the corrected
> full-book evaluation protocol. The original numbers (3.702 / 3.530) were
> inflated by a bug in `estimate_loss()` and must not be cited.

### BPC (corrected, full-book non-overlapping eval)

| Experiment | Val BPC | Test BPC | Test 95% CI | Δ vs baseline |
|------------|---------|----------|-------------|---------------|
| Random baseline | 6.570 | 6.570 | — | — |
| Baseline (const LR) | 2.179 | 2.238 | [2.225, 2.250] | — |
| Cosine LR + warmup | — | 2.234 | [2.220, 2.246] | −0.004 (n.s.) |

Paired bootstrap test (n=2000, block_len=512): p-value not significant.

### Decoding strategy results (cosine checkpoint, 50 samples per strategy)

| Strategy | TTR ↑ | 4-gram H ↑ | Rep. rate ↓ | Loop onset ↑ |
|----------|--------|-----------|------------|-------------|
| Greedy | 0.117±0.06 | 4.97±0.21 | 0.877±0.01 | 27 |
| Temp. 0.8 | 0.730±0.03 | 8.61±0.08 | 0.074±0.02 | 83 |
| Nucleus p=0.95 | 0.753±0.04 | 8.66±0.08 | 0.065±0.03 | 112 |
| Rep. penalty 1.3 | 0.838±0.03 | 8.78±0.04 | 0.033±0.01 | 75 |
| Mirostat τ=5 | 0.804±0.04 | 8.73±0.05 | 0.046±0.02 | 89 |
| No-repeat 4-gram* | 0.946±0.02 | 8.96±0.00 | 0.000±0.00 | — |

*Hard constraint — not comparable to probabilistic methods (see note above).