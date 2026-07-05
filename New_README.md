# Recurrence-Risk Decoding for Repetition-Loop Control

A decoding-time method that suppresses repetition loops in language models by
penalising candidate tokens in proportion to the risk that they would extend a
repeated n-gram. Derived as the minimum-distortion (KL-projection) solution to
a risk-bounded decoding problem:

    q(v)  proportional to  p(v) * exp( -lambda * risk(v) )

Hard no-repeat n-gram blocking is the infinite-penalty limit; the unmodified
model is the zero-penalty limit. Two configurations: **risk-only** (fixed
penalty, the core mechanism) and **adaptive** (penalty modulated online).

## Files

```
src/decoding.py        RecurrenceRiskDecoder + baselines. Risk is read from an
                       incrementally maintained hash map (context -> follower
                       set), O(N) per step, verified equal to a history scan.
sampling_eval.py       31-config evaluation: equal-budget sweeps, six loop
                       definitions, survival analysis, paired bootstrap tests,
                       leakage checks, runtime.
generalize_gpt2.py     Applies the decoder to a pretrained subword model
                       (DistilGPT-2 / GPT-2), no retraining.
plot_results.py        Pareto, KM survival, loop-robustness, ablation figures.
eval_bpc.py            Full-book BPC with paired block bootstrap (eval note).
```

## Reproduce

```bash
# Main evaluation (character model, both checkpoints)
python3 sampling_eval.py \
  --ckpt runs/baseline/best.pt runs/cosine/best.pt \
  --out_dir runs/sampling_eval_v6 --n_seeds 10 --n_chars 500

# Generalization (needs: pip install transformers; internet for first download)
python3 generalize_gpt2.py --model distilgpt2 --n_seeds 5 --n_tokens 200 \
  --out runs/generalize/distilgpt2.json

# Figures
python3 plot_results.py --dir runs/sampling_eval_v6 \
  --out runs/sampling_eval_v6/plots --generalize runs/generalize/distilgpt2.json
```

## Headline results (cosine checkpoint, 10-gram loop, 150 samples)

| Method | Survival AUC | RMST | Model-consistency NLL | 4-gram Sim | Loop % |
|--------|-------------|------|-----------------------|-----------|--------|
| Temperature 0.8 | 0.847 | 423 | 1.442 | 0.250 | 37% |
| Nucleus p=0.95 | 0.832 | 416 | 1.431 | 0.250 | 41% |
| Rep. penalty 1.3 | 0.923 | 462 | 1.727 | 0.218 | 23% |
| Mirostat τ=5 | 0.904 | 452 | 1.663 | 0.234 | 23% |
| Look-back α=3 | 0.805 | 402 | 1.293 | 0.265 | 46% |
| No-repeat 4-gram* | 1.000 | 500 | 1.800 | 0.232 | 0% |
| **RR-risk-only** | **0.975** | **487** | **1.331** | **0.270** | **3%** |
| **RR-adaptive** | 0.951 | 475 | **1.318** | **0.270** | 10% |

*Hard constraint — directly forbids the measured event; reported separately,
excluded from the Pareto frontier and significance tests.

Recurrence-risk decoding occupies the survival-quality frontier no baseline
reaches. Paired-bootstrap RMST gains over repetition penalty (+25.7, 95% CI
[+7.2, +42.8]), Mirostat (+35.4, [+15.5, +57.1]), and look-back (+85.0,
[+65.1, +107.4]) are significant, as are the NLL and similarity gains
(p < 0.001). Ablations identify the risk signal, not the online adaptation,
as the operative mechanism.

## Generalization (DistilGPT-2, subword, no retraining)

Among methods that meaningfully suppress repetition, recurrence-risk achieves
the lowest model-consistency NLL (2.68 nats vs 4.73 for repetition penalty,
3.12 for no-repeat), cutting token 3-gram repetition to ~0.015 from ~0.09 for
temperature/nucleus. The minimum-distortion property transfers across model
size and tokenisation.

## Evaluation note (not a contribution)

An earlier BPC figure (~3.70) came from a sampler covering ~27% of the test
book. Full-book evaluation gives 2.238 (constant LR) / 2.234 (cosine); the
schedule gap is real but negligible (~0.005 BPC). The lesson is recorded; the
schedule is not a subject of any claim.
