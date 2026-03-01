# Sampling Effects in Character-Level Text Generation (Baseline)

This repo contains a CPU-friendly character-level Transformer baseline and controlled sampling evaluation.

## 1) Setup
```bash
pip install -r requirements.txt



run 
```
python3 -m data.prepare \
  --out_dir data_out \
  --book_ids 1342 84 1661 98 \
  --val_books 1 --test_books 1 \
  --max_chars_per_book 2000000
```

then Sanity Check
```
python3 train.py \
  --data_dir data_out \
  --out_dir runs/overfit \
  --overfit_chars 50000 \
  --max_steps 3000
```

next full training!
```
python3 train.py \
  --data_dir data_out \
  --out_dir runs/baseline \
  --max_steps 20000
```