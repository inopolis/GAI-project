# Sampling Effects in Character-Level Text Generation

A GPT-style character-level Transformer trained on Project Gutenberg novels.  
Explores decoding strategies (greedy, temperature, top-k, nucleus) and learning rate schedules.

---

## Project Structure

```
.
├── src/
│   ├── model.py        # CharTransformerLM (GPT-style decoder-only Transformer)
│   ├── dataset.py      # CharBinDataset (memory-mapped binary file)
│   ├── decoding.py     # generate() with temperature / top-k / top-p
│   └── utils.py        # set_seed, bpc_from_loss, save/load JSON
│
├── prepare.py          # Download & preprocess Gutenberg books
├── train.py            # Baseline training (constant LR)
├── train_cosine.py     # Improved training (cosine LR + warmup)
├── sample.py           # Single-prompt text generation
├── sampling_eval.py    # Systematic evaluation of 9 decoding strategies
├── plot_training.py    # Generate training curve plots from log.csv
├── demo.py             # Quick demo — no training required
│
├── requirements.txt
└── README.md
```

---

## 1. Setup

```bash
# Clone / unzip the project, then:
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt:**
```
torch
numpy
tqdm
matplotlib
```

---

## 2. Data Preparation

Downloads 4 Project Gutenberg books and creates `data_out/` with binary-encoded train/val/test splits.

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

---

## 3. Sanity Check (Overfit Test)

Verify the model can overfit a tiny slice before full training:

```bash
python3 train.py \
  --data_dir data_out \
  --out_dir runs/overfit \
  --overfit_chars 50000 \
  --max_steps 3000
```

Expected: training loss drops to < 0.5 within 3,000 steps.

---

## 4. Full Training

### Experiment 1 — Baseline (constant LR)

```bash
python3 train.py \
  --data_dir data_out \
  --out_dir runs/baseline \
  --max_steps 20000
```

**Results:** val BPC 2.620 · test BPC 3.702 · ~13 min on Apple MPS

### Experiment 2 — Cosine LR Schedule

```bash
python3 train_cosine.py \
  --data_dir data_out \
  --out_dir runs/cosine \
  --max_steps 20000 \
  --warmup_steps 500
```

**Results:** val BPC 2.656 · test BPC **3.530** · ~13 min on Apple MPS  
→ Test BPC improved by **−0.172** compared to baseline.

---

## 5. Generate Text (Single Sample)

```bash
# Greedy decoding (deterministic, shows repetition)
python3 sample.py \
  --ckpt runs/baseline/best.pt \
  --prompt "CHAPTER 1\n" \
  --temperature 0.0

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
```

---

## 6. Quick Demo (No Training Required)

If you have a trained checkpoint, run the interactive demo:

```bash
python3 demo.py --ckpt runs/cosine/best.pt
```

Side-by-side comparison of greedy vs temperature=0.8 vs nucleus p=0.95.

---

## 7. Sampling Evaluation

Evaluates all 9 decoding strategies with TTR, 4-gram entropy, and repetition rate:

```bash
python3 sampling_eval.py \
  --ckpt runs/baseline/best.pt \
  --out_dir runs/baseline/sampling_eval
```

Output: `metrics.csv` · `sampling_metrics.png` · `samples.txt`

---

## 8. Training Plots

Generate training curve plots from any `log.csv`:

```bash
# Baseline plots
python3 plot_training.py \
  --log runs/baseline/log.csv \
  --out runs/baseline/plots

# Cosine LR plots
python3 plot_training.py \
  --log runs/cosine/log.csv \
  --out runs/cosine/plots
```

Output: `training_curves.png` · `val_bpc_curve.png` · `generalization_gap.png`

---

## 9. Hardware & Runtime

| Hardware | Speed | Full training (20k steps) |
|----------|-------|--------------------------|
| Apple M-series (MPS) | ~26 it/s | ~13 minutes |
| CPU only | ~8 it/s | ~40 minutes |
| CUDA GPU | ~80 it/s | ~4 minutes |

The model (~849K parameters) is intentionally small — it runs on CPU with no GPU required.

---

## 10. Key Results Summary

| Experiment | Val BPC | Test BPC |
|------------|---------|----------|
| Random baseline | 6.570 | 6.570 |
| Baseline (constant LR) | 2.620 | 3.702 |
| Cosine LR + warmup | 2.656 | **3.530** |

| Sampling Strategy | TTR ↑ | 4-gram Entropy ↑ | Rep Rate ↓ |
|------------------|-------|-----------------|-----------|
| Greedy | 0.147 | 5.653 | 0.784 |
| Temperature 0.8 | 0.709 | 8.856 | 0.078 |
| Temperature 1.0 | 0.784 | 8.960 | 0.053 |
| Nucleus p=0.95 | 0.732 | 8.876 | 0.071 |