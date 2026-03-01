# STAGE 2 BASELINE REPORT

---

# Baseline: Character-Level Transformer for Controlled Decoding Study

## 1. System Overview

I implemented a complete end-to-end character-level language modeling system from scratch. The system includes:

* A deterministic dataset pipeline (download → clean → split → encode)
* A decoder-only Transformer model
* A clean training script (`train.py`)
* A controlled inference script (`sample.py`)
* Logging, checkpointing, and reproducibility via fixed seeds

This baseline establishes a controlled experimental setup where only decoding methods will vary in later stages.

---

## 2. Dataset Pipeline

### Source

Public-domain English novels from Project Gutenberg.

Book IDs used:

* 1342 (Pride and Prejudice)
* 84 (Frankenstein)
* 1661 (Sherlock Holmes)
* 98 (A Tale of Two Cities)

### Preprocessing

* Stripped Project Gutenberg header and footer markers
* Normalized whitespace (collapsed repeated spaces and excessive newlines)
* Deterministic cleaning
* Character-level vocabulary built from **training split only**

### Splitting Strategy

Book-level holdout:

* Train: first N books
* Validation: 1 book
* Test: 1 book

This avoids style leakage across splits.

Vocabulary size: **95 characters**

All splits encoded into:

* `train.bin`
* `val.bin`
* `test.bin`

---

## 3. Baseline Model

Architecture:

* Decoder-only Transformer
* 4 layers
* d_model = 128
* 4 attention heads
* Context length = 256
* Dropout = 0.1

Training:

* Optimizer: AdamW
* Learning rate: 3e-4
* Gradient clipping: 1.0
* CPU training
* Fixed random seed

Objective:

* Next-character prediction using cross-entropy loss

Metric:

* **Bits Per Character (BPC)**
  [
  BPC = \frac{Loss}{\ln(2)}
  ]

---

## 4. Sanity Check (Overfit Test)

To verify correctness of the implementation, I performed an overfitting experiment on a tiny subset of 50,000 characters.

Training configuration:

* 3,000 steps
* Validation set = same subset
* CPU training

### Learning Curve

| Step | Train Loss | Val Loss | Train BPC | Val BPC |
| ---- | ---------- | -------- | --------- | ------- |
| 1    | 4.5594     | 4.2822   | 6.5779    | 6.1780  |
| 500  | 2.3669     | 2.5026   | 3.4147    | 3.6105  |
| 1000 | 2.0645     | 2.0561   | 2.9784    | 2.9664  |
| 1500 | 1.7047     | 1.6650   | 2.4593    | 2.4020  |
| 2000 | 1.4894     | 1.2685   | 2.1488    | 1.8301  |
| 2500 | 1.2592     | 0.9416   | 1.8167    | 1.3585  |
| 3000 | 1.0484     | 0.6923   | 1.5125    | 0.9988  |

Best validation loss: **0.6923**
Best validation BPC: **0.9988**

### Interpretation

The validation BPC decreases from 6.18 to 0.99, demonstrating that:

* The model can successfully fit a small dataset.
* The objective function and backpropagation are implemented correctly.
* The data pipeline produces valid training sequences.

The test BPC in this setting is high (≈5.06), which is expected since the model is trained on a very small subset and does not generalize.

This confirms that the baseline system is functioning correctly.

---

## 5. Quantitative Metric

Primary quantitative metric: **Bits Per Character (BPC)**

BPC directly measures held-out likelihood and is appropriate for character-level modeling.

Lower BPC indicates better predictive performance.

---

## 6. Qualitative Sampling

The inference script supports:

* Greedy decoding (temperature = 0)
* Temperature sampling (temperature > 0)
* Optional top-k and top-p filtering

Fixed prompts and seeds are used to ensure reproducibility.

Example prompt:

```
"CHAPTER 1\n"
```

Greedy decoding produces more repetitive but coherent output.
Temperature sampling increases diversity but may reduce local coherence.

This establishes a fair evaluation setup where only decoding strategies vary.

---

## 7. CPU-Friendly Design

To ensure efficient training:

* Small Transformer (4 layers, 128 hidden)
* Limited context length (256)
* Optional truncation of books
* Overfit sanity check on 50k characters
* CPU-only training

The system runs entirely on modest hardware.